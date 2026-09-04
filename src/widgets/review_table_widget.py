"""The gate between reading a user's table and plotting it.

Three screens, in one column: which profile describes this file, one row per column of
the file, and the two Save buttons. Nothing here decides anything -- the rules live in
`column_roles` and `profile_matching` as pure functions, because a dropdown constrains one
cell and knows nothing about the others, so anything spanning rows can only be applied
after the edit comes back.

The table edits a **session-local working copy**. A saved profile is written by
`save_working_copy` and by nothing else, which is what makes a mismatched upload
harmless: nothing is preselected, so there is no profile for it to damage.
"""
import html

import streamlit as st

from src.column_roles import (
    LABEL_ROLES,
    NO_GROUP,
    ROLE_IGNORE,
    ROLE_LABELS,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    code_span,
    column_preview,
    enforce_role_invariants,
    row_id_notice,
)
from src.dataset_io import (
    build_working_copy,
    review_blocking_reason,
)
from src.emojis import happy_emoji, sad_emoji
from src.profile_matching import (
    chooser_is_needed,
    chooser_options,
    exact_match,
    rank_profiles,
)
from src.vis.helpers import natural_key
from src.widgets.analysis_config_widgets import (
    AUTO_DETECT,
    MANAGE_LABEL,
    MAX_PROFILES,
    all_profile_columns,
    delete_profile,
    list_profiles,
    profile_roles_and_groups,
    rename_profile,
    save_working_copy,
)

# What the ungrouped slot reads on a row that *could* hold a group -- the picker the column
# will actually turn up under on the next screen. `NO_GROUP` stays the stored value; this is
# a label, not a second option. Shortened from the real group name, "Uncategorized
# Features", which does not fit the cell; the Group box's own tooltip carries that in full.
UNGROUPED_LABEL = "Uncategorized"
# The heading over the group controls. A heading rather than the sentence that used to sit
# there: the controls say what they do, and a caption explaining what a feature group *is*
# re-explained it on every render of every file. The explanation is worth having once, so it
# rides on the heading's own tooltip -- hover, not a line, which is the whole difference.
GROUP_SECTION = "Feature group management"
GROUP_HELP = ("Each group becomes one dropdown in the feature pickers, so features are "
              "chosen from a few short lists instead of one long one. Measurements left "
              "ungrouped are collected under Uncategorized Features.")

# Cleared together whenever a different file arrives, so no edit can outlive the table it
# was made against.
_STATE_KEYS = (
    "_review_fingerprint", "_review_roles", "_review_groups", "_review_group_names",
    "_review_source", "_review_confirmed", "_review_previous_roles", "_review_known_cols",
    "_review_notices", "_review_opened", "_review_saved_as", "_review_numeric_cols",
    "_review_reopened", "_review_chooser", "_review_delete_armed", "_review_manage_open",
    "_review_configured_row_id", "_review_picks_stale",
)
# The name the table gave the Row ID role, blank included -- read by the export button,
# which needs the *configured* answer so the generated script re-invents the same
# "Row number" column instead of demanding one the data file never had. Written and
# cleared here rather than by the page, so its lifetime is the uploaded file's like every
# other key above: written from a different module it was the one `_review_*` key a new
# upload could inherit.
_CONFIGURED_ROW_ID = "_review_configured_row_id"
# Which profile's row is showing its delete confirm, if any. One name, so arming a second
# row disarms the first for free, and a rerun after the delete finds nothing armed.
_DELETE_ARMED = "_review_delete_armed"
# Whether the manage section is open. Set by every control inside it -- see _manage_profiles.
_MANAGE_OPEN = "_review_manage_open"
# One row of the table: the tick, the new badge, the column's name, its two dropdowns, a
# preview. The tick's slot is carved out of the preview rather than added to the total, so
# the badge keeps the width it was measured at inside the page's own narrow-ish column.
# Group is a slot wider than Role even though its options are shorter: the *ungrouped*
# slot reads UNGROUPED_LABEL, which is longer than any role, and at Role's width it
# truncated to "Uncateg..." -- less informative than the dash it was meant to replace.
_ROW_WEIGHTS = (1, 1, 5, 3, 4, 4)
# Measured off the rendered table: a row is 56px of selectbox and gap, and the header
# plus the box's own padding is 78. Under-guess either and a four-column file scrolls a
# box it fits in. Past ~7 columns the list scrolls rather than pushing Save down the page.
_ROW_HEIGHT, _HEADER_HEIGHT, _MAX_TABLE_HEIGHT = 56, 78, 470

_GEN = "_review_editor_gen"
# A second counter, bumped only when a different file arrives. The editor's generation
# moves on every corrected edit, so keying the chooser to it would throw away the user's
# profile pick mid-table.
_FILE_GEN = "_review_file_gen"


def _fingerprint(uploaded_file, df):
    """What makes this the same table as last run: the name and the column set.

    Not the contents: a rerun must not rebuild the working copy and discard the user's
    edits, and a genuinely different upload always differs in one of these two.
    """
    return (getattr(uploaded_file, "name", ""), tuple(df.columns))


def _file_gen():
    return st.session_state.get(_FILE_GEN, 0)


def _table_height(columns):
    """Tall enough for the rows, capped at a box the Save button still fits under."""
    return min(_MAX_TABLE_HEIGHT, _HEADER_HEIGHT + _ROW_HEIGHT * columns)


def _bump_editor():
    """Re-key every row so the table re-reads the working copy.

    A delete-key passes AppTest and fails in a browser -- the frontend restores the old
    value on rerun -- and writing to a widget's key after it has rendered raises. The
    generation counter is the only way to move a value into a row from code below it.
    """
    st.session_state[_GEN] = st.session_state.get(_GEN, 0) + 1


def _group_label(name):
    """The ungrouped slot, named after where the column really goes.

    A `format_func` rather than a second option, because the options list has to stay the
    same on every row: vary it by role and a column promoted to Numerical hands its
    selectbox a value the new list no longer offers, which Streamlit refuses outright.

    Only measurements get it. On a Categorical, Row ID or Ignore row the dash is the honest
    answer -- a group does not apply there, so the column is not *un*-grouped so much as
    out of scope, and naming a destination it can never reach would only invite the click
    the box is disabled to prevent.
    """
    return UNGROUPED_LABEL if name == NO_GROUP else name


def _group_key(gen, col, measurement):
    """The key of one row's Group box, which carries the *label mode* it is showing.

    **A rendered selectbox's option labels are fixed at its key.** Flipping `format_func`
    on the next run repaints nothing: the box that said `Uncategorized` as a measurement
    went on saying it after the row was demoted to Categorical, and `—` went on showing
    after a promotion. Measured on Streamlit 1.54 with three boxes side by side -- only
    `format_func` changing, `format_func` and `disabled` changing, and the key changing --
    and only the third repainted. So the mode joins the identity, and the re-key is what
    makes `_group_label` visible at all.

    `AppTest` reports the fresh label either way, which is why the test for this asserts
    the key rather than the text -- the same shape as the delete-key rule in the root
    `CLAUDE.md`, where the harness also disagrees with the browser.

    The stale value under the old key is deliberate: demote a grouped column and promote it
    back and its group returns, because `enforce_role_invariants` strips the group on the
    way out and the box remembers it on the way in. Nothing lies -- the row's label and
    `_review_groups` agree either way.
    """
    return f"review_group_{gen}_{'num' if measurement else 'other'}_{col}"


def _pick_key(col):
    """The key of one row's tick.

    Keyed to the *file*, not the editor: `_GEN` moves on every corrected edit, and the
    whole point of ticking twelve rows is to act on them afterwards, so dropping the
    selection because an unrelated Row ID was demoted is exactly the loss `_FILE_GEN` was
    split out to prevent for the name boxes. A new file is the one thing that must forget
    them -- two files can share a column name, and its tick would ride across.
    """
    # `review_tick_`, not `review_pick_`: the chooser's rows are `review_pick_{profile}`,
    # so a profile named after a column -- or after `all` -- would collide with a key here.
    return f"review_tick_{_file_gen()}_{col}"


def _picked_columns(df):
    """The ticked measurements, in the file's own column order.

    Read out of session state above the rows that hold the boxes: Streamlit applies widget
    state before the script body, so the bar sees the tick the user just made rather than
    last run's count. Filtered by role because a row demoted this run keeps its tick for
    one rerun -- the widget is gone, its key is not.
    """
    roles = st.session_state._review_roles
    return [col for col in df.columns
            if roles.get(col) == ROLE_NUMERICAL
            and st.session_state.get(_pick_key(col), False)]


def _clear_picks(columns):
    """Untick, by assignment rather than by deleting the key.

    Every caller runs above the rows, so the boxes have not instantiated yet and a plain
    write lands -- which is what makes the ticks the one widget in this file needing no
    generation counter of their own. Deleting instead would pass `AppTest` and fail in a
    browser: the frontend re-sends a value the server no longer holds.
    """
    for col in columns:
        st.session_state[_pick_key(col)] = False


def _load_working_copy(df, picked):
    """Build the table's contents from a chooser pick, discarding any edits in flight."""
    source = None if picked == AUTO_DETECT else picked
    profile_roles, profile_groups, profile_names = ({}, {}, [])
    if source:
        profile_roles, profile_groups, profile_names = profile_roles_and_groups(source)
    roles, groups, numeric_cols = build_working_copy(
        df, profile_roles, profile_groups, profile_group_names=profile_names)
    st.session_state._review_roles = roles
    st.session_state._review_groups = groups
    # The profile's own group order first -- apply_column_groups fixes that order and the
    # feature pickers show it, so re-deriving it from column order would reshuffle the
    # pickers on every save. Read from the profile's stored group *names*, not from the
    # groups this file's columns landed in: a group the profile has but this file cannot
    # fill has no column here to be read off, and it is still a name the user chose --
    # reading it off the mapping is what made apply_column_groups' group_names argument
    # half-dead, saving an empty group that the next upload of the same file dropped.
    names = list(dict.fromkeys(profile_names))
    for col in df.columns:
        group = groups.get(col)
        if group and group not in names:
            names.append(group)
    st.session_state._review_group_names = names
    st.session_state._review_source = picked
    # These roles came from somewhere else now, so the name an earlier Save gave them no
    # longer describes what the table holds.
    st.session_state.pop("_review_saved_as", None)
    st.session_state._review_previous_roles = dict(roles)
    st.session_state._review_known_cols = set(profile_roles)
    # A fact about the frame, so it is stored once per file rather than recomputed per
    # keystroke: the 1% coercion behind it re-reads every text column, which on a wide
    # table of free text dominates the load (~62 ms on 50k x 20, against ~11 ms for all
    # the previews). It comes back from build_working_copy, which coerced a copy to guess
    # the roles -- asking for it separately walks the frame a second time for the same
    # answer, and did: ~150 ms for the pair against ~90 ms for the one call.
    st.session_state._review_numeric_cols = numeric_cols
    # Not cleared here: `Cancel` reaches this from *below* the rows, where writing a
    # rendered widget's key raises. `_group_section` does it on the next run, above them.
    st.session_state._review_picks_stale = True
    _bump_editor()


def _applied_profile():
    """The saved profile the loaded file is using, or None for an auto-detected copy.

    A Save gives the working copy a name, and that name outranks the chooser pick:
    reporting "Auto-detected" after an explicit Save would hide the one thing the user
    just did.

    Deliberately not `current_profile`, which only a Save, a rename or a delete sets --
    on an exact match it still names whichever profile was written last, which need not
    be the one this file matched. Every surface that says which profile is in force reads
    this instead, so they cannot disagree.
    """
    source = (st.session_state.get("_review_saved_as")
              or st.session_state.get("_review_source"))
    return None if source in (None, AUTO_DETECT) else source


def applied_summary(decision):
    """One line naming the profile in force and the roles it gave this file.

    The only evidence a profile was applied. An exact match renders no gate at all --
    that is the point of it -- but being silent about *which* profile leaves the user
    unable to tell a match on `pdl1` from one on `pdl1-rep3`, both of which are a
    plausible fit for the file in front of them. It also has to name the auto-detected
    case, where there is no profile: "Auto-detected" is a real answer, not a missing one.

    Counted from `decision["roles"]`, which is one entry per column *of the file*, so the
    line describes what was decided rather than what survived the prune -- a column
    dropped for being empty was still assigned a role. Ordered by ROLE_LABELS, which is
    the roles' own precedence order, so the identifiers come before the measurements.
    """
    roles = decision.get("roles") or {}
    if not roles:
        return
    counts = dict.fromkeys(ROLE_LABELS, 0)
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1
    # Row ID is left out: it is one column or none, so counting it says nothing.
    # What the line is for is the shape of what the pickers below are handed.
    tally = " · ".join(f"{n} {ROLE_LABELS[role]}" for role, n in counts.items()
                       if n and role != ROLE_ROW_ID)
    name = _applied_profile() or "Auto-detected"
    # The ✏️ rides on this line because the line is the only thing an exact match
    # renders: the gate it reopens never appeared, so there is nowhere else to hang it.
    # A horizontal container, not st.columns: columns split the row by ratio, so the
    # glyph sat at the far right of the upload column with the sentence's tail nowhere
    # near it. Both children are content-width and the row is pinned `nowrap` in the
    # <style> below, which is what a plain horizontal container would not give -- it
    # wraps, and the pencil dropped to a line of its own in a column this narrow.
    with st.container(key="review_summary", horizontal=True,
                      vertical_alignment="center", gap=None):
        # Hand-drawn rather than st.caption so the <style> can ride in the same markdown
        # call. A button is ~2.5rem tall against a caption's ~1.5, and a row that tall
        # left this one-liner marooned between the upload message and the picker; the
        # negative margin then takes back half of what the vertical block puts above it,
        # because this line belongs to the upload above rather than the picker below.
        # A separate <style> element would undo both: one more child of a gapped block
        # adds back the space it removes -- the reason _picker_label inlines its own.
        # The name is a profile's, typed by the user, and this renders as markup, so it
        # is escaped here at the interpolation point; the <b> is ours and deliberate.
        st.markdown(
            # nowrap keeps the glyph on the sentence's line; the pencil is told not to
            # shrink so a long tally squeezes the text and never the target.
            "<style>.st-key-review_summary{margin-top:-0.5rem;flex-wrap:nowrap}"
            # Streamlit's markdown wrapper carries margin-bottom:-16px, which collapses
            # a paragraph's trailing space -- and here made a 24px line box measure 8,
            # so the sentence centred 8px below the glyph beside it. Every wrapper
            # between the row and the text is pinned to the line's own height, and that
            # margin taken back; measured in the browser, and nothing re-checks it.
            ".st-key-review_summary>div:first-child,.st-key-review_summary .stMarkdown,"
            ".st-key-review_summary .stMarkdown div"
            "{height:1.5rem;display:flex;align-items:center;margin-bottom:0}"
            ".st-key-review_reopen{flex:0 0 auto}"
            ".st-key-review_reopen button{min-height:0;height:1.5rem;padding:0 0.4rem}"
            ".st-key-review_reopen p{line-height:1.5rem}</style>"
            "<div style='font-size:0.875rem;opacity:0.65;margin:0;line-height:1.5rem;"
            f"white-space:nowrap'><b>{html.escape(name)}</b>:&nbsp;{tally}</div>",
            unsafe_allow_html=True, width="content")
        # Tertiary: plain text, no box. A boxed button beside a one-line caption reads
        # as the row's main event, and this row's main event is the sentence. The
        # padding above is what keeps a ~20px glyph clickable once the box is gone.
        if st.button("✏️", key="review_reopen", type="tertiary",
                     help="Review this file's columns again"):
            reopen_gate()
            st.rerun()


def _decision():
    roles = st.session_state.get("_review_roles", {})
    # Recorded on the way out, because the export button reads it many frames later and
    # cannot re-derive it: by then the frame carries the *resolved* identifier, which for
    # a table with none is an invented column. `apply_column_roles` picks the same first
    # entry, so a Save and this cannot disagree about which column holds the role.
    st.session_state[_CONFIGURED_ROW_ID] = next(
        (col for col, role in roles.items() if role == ROLE_ROW_ID), "")
    return {
        "profile": _applied_profile(),
        "roles": roles,
        "groups": st.session_state.get("_review_groups", {}),
        "group_names": st.session_state.get("_review_group_names", []),
    }


def configured_row_id():
    """The column the review table marked Row ID, or "" when the table names none.

    The **configured** answer, never the resolved one: blank means the exported script
    must invent its own row numbers, and baking in the invented name would make
    `check_and_fix_df` demand a column the data file never had. The reverse of what the
    FOV name does, which is passed to the export *resolved* -- see the root `CLAUDE.md`.
    """
    return st.session_state.get(_CONFIGURED_ROW_ID, "")


def ignored_columns():
    """The columns the review table marked Ignore.

    Read by the export button, because `coerce_majority_numeric_cols` takes a set of
    columns to *skip*: an ignored column left out of the script's set is converted there
    and not in the app, so the script reports a conversion the user never saw -- for a
    column they had dismissed -- and on the `ANALYSIS_COLUMNS = None` path keeps a
    numeric column the app's frame never held.

    Derived from `_review_roles` rather than snapshotted like the Row ID: that key is
    already the widget's own, cleared when a different file arrives, so there is no
    second lifetime to keep in step. The Row ID needs the snapshot because by export
    time the frame carries the *resolved* identifier; an ignored column has no resolved
    form -- it is simply gone.
    """
    roles = st.session_state.get("_review_roles", {})
    return [col for col, role in roles.items() if role == ROLE_IGNORE]


def review_gate(uploaded_file, df):
    """Show the gate, or nothing at all. Returns the decision once confirmed, else None.

    An exact match auto-applies and this renders nothing -- the plot page's summary bar is
    then the only evidence a profile was applied. Auto-apply is an *entry* decision, taken
    once per file: pressing Save inside the gate makes the profile match the file exactly,
    and without `_review_opened` that would slam the table shut mid-edit.

    **One read of `analysis_config.toml` per run at most, threaded down.** Every place
    below that wants the saved profiles is handed `profiles`, computed lazily: three
    separate calls cost three uncached TOML parses of every saved profile (2.7 ms each at
    the cap), on a page Streamlit reruns on every slider drag. Lazily, because the
    confirmed path returns above them all and must read nothing. Passing the value down
    rather than memoising it deliberately: the gate writes the config from inside a rerun,
    so a stale profile list would be a worse bug than a slow one.
    """
    profiles = None
    fingerprint = _fingerprint(uploaded_file, df)
    if st.session_state.get("_review_fingerprint") != fingerprint:
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        st.session_state._review_fingerprint = fingerprint
        # Re-key rather than delete-key: a keyed radio keeps its value across reruns, so
        # without this the second file arrives with the first file's profile already
        # applied -- and a "Save to <that profile>" button one click from overwriting it
        # with a column set its owner never chose.
        st.session_state[_FILE_GEN] = _file_gen() + 1
        _bump_editor()

    if not st.session_state.get("_review_confirmed") and not st.session_state.get("_review_opened"):
        profiles = all_profile_columns()
        matched = exact_match(set(df.columns), profiles)
        if matched:
            _load_working_copy(df, matched)
            # Matching the columns is not the same as still working on them. A profile
            # remembers which column is the identifier and never whether it holds
            # anything, so an export that leaves it blank is still an exact match --
            # and applying it silently sent the file to interpret_table to fail there,
            # with no table ever shown. Ask the gate's own question before stepping
            # aside; when it has an answer, the table opens on it instead.
            st.session_state._review_confirmed = not review_blocking_reason(
                df, st.session_state._review_roles)

    if st.session_state.get("_review_confirmed"):
        return _decision()

    st.session_state._review_opened = True
    # Decided once per opening, not per run: in the legacy state where two profiles hold
    # the same columns, picking one would make the rule flip to False and the list would
    # vanish on the very click that chose it.
    if "_review_chooser" not in st.session_state:
        if profiles is None:
            profiles = all_profile_columns()
        st.session_state._review_chooser = chooser_is_needed(
            _applied_profile(), set(df.columns), profiles)
    _render_gate(uploaded_file, df, profiles)
    return None


# ------------------------------------------------------------------------ the screens

def _render_gate(uploaded_file, df, profiles=None):
    """The three screens, plus the manage list, off **one** read of the saved profiles.

    `profiles` is `{name: known columns}` -- whatever the caller already had in hand.
    The chooser needs the columns; the manage list and the `Save as` count need only the
    names, which are its keys, so one read serves all three. `list_profiles()` is the
    cheaper read when the chooser is suppressed (the ✏️ path), where the columns go
    unused.
    """
    name = getattr(uploaded_file, "name", "the uploaded file")
    st.caption(f"Read {len(df.columns)} columns × {len(df):,} rows from {name}")

    if st.session_state.get("_review_chooser", True):
        if profiles is None:
            profiles = all_profile_columns()
        saved_names = list(profiles)
        _chooser(df, profiles)
        # The pick lives in `_review_source` -- the key `_load_working_copy` writes --
        # rather than in a widget, so there is nothing to compare here: a row that was
        # clicked has already loaded itself.
        if st.session_state.get("_review_source") is None:
            return
    else:
        saved_names = list(profiles) if profiles is not None else list_profiles()

    for notice in st.session_state.pop("_review_notices", []):
        st.info(notice)
    _group_section(df)
    _editor(df)
    # Directly under the rows, because the Role column is what decides it. Read from
    # session state so it describes the *corrected* roles, and said at all because the
    # numbering is invented after the gate closes: nothing downstream announces it.
    numbering = row_id_notice(st.session_state._review_roles)
    if numbering:
        st.info(numbering)
    _buttons(df, saved_names)
    _manage_profiles(saved_names)


def _chooser(df, profiles):
    """Rank the saved profiles against this file, one row each. Orders the list; picks nothing.

    Candidates only: `fits` comes from `rank_profiles`, which hides nothing, while the
    rows come from `chooser_options`, which drops every profile sharing no column with
    this file. Keying the lookup off the wider of the two is what lets that cutoff move
    without this function noticing.

    Nothing here renames or deletes -- those live in `_manage_profiles`, which lists every
    profile because managing is not choosing, and because this list hides the ones sharing
    no column with the file. One question per screen is the point: this one asks which
    profile the file *is*, and should offer only answers.

    The rows stay buttons rather than becoming a radio group, which the split would
    otherwise allow: the pick is not a widget value. `_load_working_copy` owns
    `_review_source` and writes it from code below the list, which Streamlit forbids on
    a key a radio has already rendered.
    """
    file_cols = set(df.columns)
    fits = {fit.name: fit for fit in rank_profiles(file_cols, profiles)}
    options = chooser_options(file_cols, profiles, AUTO_DETECT)
    picked = st.session_state.get("_review_source")

    st.markdown("**Which profile describes this file?**")
    for option in options:
        if option == AUTO_DETECT:
            _pick_row(df, option, option, picked)
            continue
        fit = fits[option]
        _pick_row(df, option, f"{option}  —  {len(fit.shared)} shared · "
                  f"{len(fit.missing)} missing · {len(fit.new)} new", picked)

    if picked is None:
        # Only ever names the next thing on screen. The table and the button row are
        # both below the early return, so a caption pointing at either of them --
        # "press **Use this →**", say -- points at empty space.
        st.caption("Ranked by how many of this file's columns each profile already "
                   "knows; profiles sharing none are left out. Pick one to fill in "
                   "the table below.")
    elif len(options) > 1:
        st.caption("Changing this rebuilds the table below and discards your edits.")


def _pick_row(df, name, label, picked):
    """One row of the list, and the click that loads it.

    Primary for the row in force, tertiary -- plain text, no box -- for the rest, so the
    list reads as a list rather than a toolbar. Content width, not stretch: Streamlit
    centres a button's label, so a stretched row hangs its name in the middle of the row
    and the highlight becomes a full-width red bar. The label is a sentence long, so the
    hit area is generous either way. The rerun is what repaints that highlight: the rows
    above this one have already drawn themselves against the old pick.
    """
    if st.button(label, key=f"review_pick_{name}",
                 type="primary" if name == picked else "tertiary") and name != picked:
        _load_working_copy(df, name)
        st.rerun()


def _manage_profiles(saved_names):
    """Rename and delete, for *every* saved profile. The only screen that can reach them all.

    Separate from the chooser, which lists only the profiles sharing a column with the
    file in hand: renaming and deleting are maintenance and want the whole list, or a
    profile unrelated to the file is unreachable. Sorted by name, not by fit -- a fit is
    an answer to the chooser's question, and this one is not asking it.

    Placed below the button row because the screen's actual question is the table, and
    because the at-the-cap message points here: `_save_as_new` is the row directly above,
    so "below" is true whenever that error can appear.

    It renders on *every* opening, the ✏️ reopen of an exact match included, where
    `chooser_is_needed` suppresses the chooser entirely. That is the only reason a user
    whose files all auto-apply can prune at all.

    Collapsed by default, and that saves nothing at render time -- Streamlit runs an
    expander's body whether it is open or not. What it buys is that maintenance does not
    compete with the table for the eye.

    **The open state is ours, not the frontend's.** A delete removes a row, which remounts
    the expander, and `expanded` only initialises -- so without `_MANAGE_OPEN` the panel
    closes on the click that pruned one profile, in the middle of pruning several. Every
    control inside sets the flag, and the remount reads it. It does not pin the panel
    open, since a plain rerun does not remount, so a manual collapse still sticks. The
    flag is in `_STATE_KEYS`, so a new file starts closed.

    The label carries no count, tempting as `(4 of 20)` is: a changing label is a second
    reason to remount, and the count already sits on the `Save as` box's tooltip, which
    is where the cap actually bites.

    **Every row is a keyed container**, and that is the same problem one level down.
    `st.popover` and `st.button` take their identity from position when nothing else
    distinguishes them, so deleting the second of four hands the third that slot and its
    frontend state with it -- an armed control appearing unasked, under the cursor, naming
    a profile the user had not chosen. Keying the container on the profile's own name is
    what makes a row's identity follow the profile rather than the slot.

    `saved_names` comes from the caller's single read of the config -- see `_render_gate`.
    """
    names = sorted(saved_names, key=natural_key)
    if not names:
        return
    with st.expander(MANAGE_LABEL, expanded=st.session_state.get(_MANAGE_OPEN, False)):
        for name in names:
            with st.container(key=f"review_manage_row_{name}"):
                if st.session_state.get(_DELETE_ARMED) == name:
                    _delete_confirm(name)
                else:
                    _manage_row(name)


def _manage_row(name):
    cols = st.columns([12, 1, 1], vertical_alignment="center")
    with cols[0]:
        # `st.text`, not `st.markdown`: a profile the user called `*draft*` is a name, not
        # an instruction to render it in italics.
        st.text(name)
    with cols[1]:
        _rename_row(name)
    with cols[2]:
        if st.button("🗑️", key=f"review_arm_delete_{name}",
                     help=f"Delete {name}", width="stretch"):
            st.session_state[_DELETE_ARMED] = name
            st.session_state[_MANAGE_OPEN] = True
            st.rerun()


def _rename_row(name):
    """A form inside the popover, so Return in the name field renames.

    Without one, Return reruns the script and closes the popover with nothing written --
    which is why the row's box needs the form and the save-as box beside `Save as` does
    not: that one is not inside anything that a rerun can close.
    """
    with st.popover("✏️", help=f"Rename {name}", width="stretch"), st.form(
            f"review_rename_form_{name}", border=False):
        # Re-keyed per file rather than per rename: a name typed and never submitted
        # would otherwise be restored by the frontend over the next file's row.
        new_name = st.text_input("Rename to", value=name,
                                 key=f"review_rename_{_file_gen()}_{name}")
        if st.form_submit_button("Rename", key=f"review_rename_submit_{name}"):
            _rename_and_refresh(name, new_name)


def _delete_confirm(name):
    """The row, replaced in place by its own confirm. What arms it is `_DELETE_ARMED`.

    Confirmed at all because what it destroys is a review the user sat through once: the
    roles and groups for every column of that file, recoverable only by uploading it again.

    **Server state, never an `st.popover`.** A popover keeps its open state in the browser
    and takes no key, so its identity is its slot: after a delete, the row that slides up
    into the freed slot inherits an open confirm the user never asked for, one click from
    destroying a profile they had not named. Session state cannot do that -- the armed
    name is *the profile*, only one row can hold it, and `_delete_and_refresh` clears it,
    so a rerun always lands on a row armed for exactly what the user clicked, or none.
    """
    cols = st.columns([8, 2, 2], vertical_alignment="center")
    with cols[0]:
        st.text(f"Delete {name}? Its roles and groups go with it.")
    with cols[1]:
        if st.button("Delete", key=f"review_delete_{name}", type="primary", width="stretch"):
            _delete_and_refresh(name)
    with cols[2]:
        if st.button("Keep", key=f"review_delete_cancel_{name}", width="stretch"):
            st.session_state.pop(_DELETE_ARMED, None)
            st.session_state[_MANAGE_OPEN] = True
            st.rerun()


def _rename_and_refresh(old, new):
    """The working copy is bound to its profile by name, so the name has to follow.

    Two keys hold it -- the pick that loaded the copy and the name a Save gave it -- and
    leaving either behind points the gate's one write at a profile that no longer exists.
    """
    error = rename_profile(old, new)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    for key in ("_review_source", "_review_saved_as"):
        if st.session_state.get(key) == old:
            st.session_state[key] = (new or "").strip()
    st.session_state[_MANAGE_OPEN] = True
    st.rerun()


def _delete_and_refresh(name):
    """Deleting the profile in force puts the gate back to asking which profile this is.

    The working copy outlives its profile otherwise: the table stays open on roles with
    nowhere to be written back to, under a button offering to save to a name that is gone.

    **Orphaned is `_applied_profile`'s question, not `_review_source`'s.** Those two keys
    disagree by design: a pick writes `_review_source` and a Save writes `_review_saved_as`,
    and `Save as` from `Auto-detect` leaves the first on the literal AUTO_DETECT label while
    the second holds the new profile's name -- so on the commonest route of all (create a
    profile, reopen it, delete it) the pick names no profile at all. The predicate has to be
    the same one the Save button reads, or the button offers a write the delete has already
    made impossible. `_review_source` is checked too, so a stale pick cannot be left naming
    a profile that is gone.

    Three keys then have to move together, and two are only reachable this way.
    `_review_chooser` is normally decided **once per opening** -- so that a pick cannot make
    the list vanish under the cursor on the click that chose it -- but a delete is not a
    pick, and on the ✏️ path that flag is False precisely because the profile now being
    deleted described the file exactly. Left alone, the gate renders a table whose roles
    have just been thrown away and never asks the question again. And `_review_reopened`
    has to go with it: it means "there is a previous decision to fall back to", which stops
    being true the moment that decision is deleted -- and its Cancel, on a copy that then
    came from nowhere, would reload the deleted profile as an empty config, auto-detect the
    roles and confirm them straight through to the plots, unsaved, which is the one thing
    the gate exists to prevent.
    """
    error = delete_profile(name)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    # Disarmed before the rerun, or the row that slides up into the freed slot renders
    # a confirm the user never asked for -- the slot-identity problem, in state this time.
    st.session_state.pop(_DELETE_ARMED, None)
    # A row fewer remounts the expander, so the flag is what keeps it open to prune again.
    st.session_state[_MANAGE_OPEN] = True
    if name in (_applied_profile(), st.session_state.get("_review_source")):
        for key in ("_review_source", "_review_saved_as", "_review_roles", "_review_groups",
                    "_review_group_names", "_review_known_cols", "_review_previous_roles",
                    "_review_reopened"):
            st.session_state.pop(key, None)
        st.session_state._review_chooser = True
    st.rerun()


def _editor(df):
    """One row per column of the file, and the two dropdowns that decide its fate.

    Real widgets rather than `st.data_editor`: a SelectboxColumn cell opens on the
    *second* click -- the first only selects it -- which is three clicks to change one
    role, on the screen whose entire purpose is changing roles. Measured at 0.22s per
    rerun on a 200-column file against the grid's 0.00s, which buys back the affordance
    and, with it, an editing path `AppTest` can actually drive.

    The name and the preview are `st.text`: both are file content, and `st.markdown`
    would render a column called `*note*` in italics and one called `# id` as a heading.
    The badge beside them is ours, so it may be markup.

    The rows sit in a fixed-height scroll box. Without one a 200-column file is a
    200-row page and the Save button is nowhere near the table it belongs to.
    """
    roles = st.session_state._review_roles
    groups = st.session_state._review_groups
    known = st.session_state.get("_review_known_cols") or set()
    role_options = list(ROLE_LABELS.values())
    group_options = [NO_GROUP] + st.session_state._review_group_names
    gen = st.session_state[_GEN]

    edited_roles, edited_groups = {}, {}
    with st.container(height=_table_height(len(df.columns)), border=True):
        header = st.columns(_ROW_WEIGHTS, vertical_alignment="center")
        for slot, title in zip(header[2:], ("Column", "Role", "Group", "Preview")):
            slot.caption(f"**{title}**")
        for col in df.columns:
            cells = st.columns(_ROW_WEIGHTS, vertical_alignment="center")
            role = cells[3].selectbox(
                f"Role of {col}", role_options, key=f"review_role_{gen}_{col}",
                index=role_options.index(ROLE_LABELS[roles[col]]),
                label_visibility="collapsed")
            held = groups.get(col, NO_GROUP)
            measurement = LABEL_ROLES[role] == ROLE_NUMERICAL
            group = cells[4].selectbox(
                f"Group of {col}", group_options,
                # Keyed to the label mode, not just the row -- see `_group_key`.
                key=_group_key(gen, col, measurement),
                index=group_options.index(held) if held in group_options else 0,
                label_visibility="collapsed",
                # Reads "Uncategorized" here and "—" on every other role -- see
                # `_group_label`. The value behind both spellings is `NO_GROUP`.
                format_func=_group_label if measurement else str,
                # Said by the control rather than by a rule that fires afterwards. The
                # rule still runs -- a role changed *this* run leaves the box behind.
                disabled=not measurement,
                # The full group name lives here, since it is too long for the cell and
                # there is no caption under the table saying it any more.
                help="Only a Numerical column can sit in a group. Ungrouped measurements "
                     "fall to Uncategorized Features.")
            # Placed first, read last: a group is a measurement's to hold, so the tick has
            # to follow the role the row is *showing*, not the one last run enforced. Slots
            # are containers, so screen order and code order are free to disagree -- the
            # same trick `_save_as_new` uses on its name box. Absent rather than disabled
            # on the other roles: a tick that cannot be acted on is noise, and the Group
            # box one cell over already says why.
            if measurement:
                cells[0].checkbox(f"Select {col}", key=_pick_key(col),
                                  label_visibility="collapsed",
                                  help="Tick to include in a bulk group assignment")
            # Why a row's role is a guess rather than something the profile stored.
            if known and col not in known:
                cells[1].markdown(":orange-badge[new]")
            cells[2].text(col)
            cells[5].text(column_preview(df[col]))
            edited_roles[col] = LABEL_ROLES[role]
            if group != NO_GROUP:
                edited_groups[col] = group

    fixed_roles, fixed_groups, notices = enforce_role_invariants(
        edited_roles, edited_groups,
        numeric_cols=st.session_state.get("_review_numeric_cols") or set(),
        previous_roles=st.session_state.get("_review_previous_roles"),
    )
    st.session_state._review_roles = fixed_roles
    st.session_state._review_groups = fixed_groups
    st.session_state._review_previous_roles = dict(fixed_roles)
    if fixed_roles != edited_roles or fixed_groups != edited_groups:
        # The correction has to be pushed back into the rows, which only re-read their
        # value under a new key. The notices survive the rerun in session state.
        st.session_state._review_notices = notices
        _bump_editor()
        st.rerun()


def _group_section(df):
    """Everything about feature groups, in one headed section above the table.

    Two rows, in the order the work happens. Row one **makes** a group -- a name box and
    Add, and nothing else, because creating one inside the Apply button read backwards:
    there is nowhere to assign to until the group exists. Row two **works on** one: the
    selectbox, then every verb that takes it as their subject -- Delete, Rename, and Apply
    filling it from the ticked rows.

    Delete and Rename sat in row one until it was pointed out that they act on row two's
    selection. That is the same objection that moved the whole section above the table:
    a control a scroll -- or a row -- away from the thing it acts on reads as unrelated to
    it. The name box is the one thing that stays put, directly above the selectbox, and
    serves Add and Rename both, because both want the same thing: a name for a group.

    So the selectbox is read where it renders, and only the *prune* above it has to run
    early.

    What the bulk half buys, stated plainly: it does not beat N *actions* -- N ticks plus
    a destination plus Apply ties with N per-row dropdowns -- but a tick is one click
    where a dropdown is two, the destination is chosen once instead of N times, and the
    choice is visible before it is committed. Misfiling the fifth of eight rows inside a
    scroll box stays invisible until the feature pickers come up wrong.

    Above the rows, and not for looks. Everything here writes the ticks' keys, which is
    only legal before they instantiate -- the same reason the ticks need no generation
    counter of their own.
    """
    # Deferred from `_load_working_copy`, which a Cancel reaches from below the rows.
    if st.session_state.pop("_review_picks_stale", False):
        _clear_picks(df.columns)
    roles = st.session_state._review_roles
    numeric = [col for col in df.columns if roles.get(col) == ROLE_NUMERICAL]
    if not numeric:
        # Nothing can hold a group, and `review_blocking_reason` is already saying so.
        return
    picked = _picked_columns(df)
    names = st.session_state._review_group_names
    gen = st.session_state[_GEN]

    # Pruned before the widget renders, which is the whole reason the selection is read
    # out of session state at all: a group can be deleted or renamed out from under this
    # key, and a keyed widget handed a value it no longer offers raises. Streamlit applies
    # widget state before the script body, so the current pick is already here to check.
    # Same idiom as `encoding_state.prune_to_options` and `filter_widgets.resolve_selections`.
    options = [NO_GROUP] + names
    target_key = f"review_group_target_{gen}"
    if st.session_state.get(target_key, NO_GROUP) not in options:
        st.session_state[target_key] = NO_GROUP

    st.markdown(f"**{GROUP_SECTION}**", help=GROUP_HELP)
    # Two narrow slots for the emoji verbs, and the same slot count and total on both
    # rows, so the name box stays exactly above the group it renames -- Streamlit sizes a
    # column against the width left after the gaps, so a row of five and a row of four
    # would not line up even on equal weights.
    create = st.columns([3, 2, 1, 1, 2], vertical_alignment="center")
    act = st.columns([3, 1, 1, 2, 2], vertical_alignment="center")

    new_name = create[0].text_input(
        "Group name", key=f"review_group_name_{gen}", label_visibility="collapsed",
        placeholder="group name")
    if create[1].button("➕ Add", key="review_add_group", width="stretch",
                        help="Make an empty group with the name on the left"):
        _add_group(new_name)

    # No `index=`: the key carries the selection, and passing both makes Streamlit warn
    # that the widget "was created with a default value but also had its value set via the
    # Session State API" -- on screen, in a yellow box, every time Add seeds it. Absent a
    # stored value the default is slot 0, which is NO_GROUP, which is what it should be.
    target = act[0].selectbox(
        "Group", options, key=target_key, label_visibility="collapsed",
        # Always a measurement's destination, so always the informative label.
        format_func=_group_label,
        help=f"The group Delete, Rename and Apply act on. {UNGROUPED_LABEL} is not a "
             "group: applying it takes the ticked rows out of theirs.")
    # Disabled until there is a group to act on, rather than hidden: a section whose
    # controls come and go reads as one that broke. `target` is NO_GROUP until then, and
    # neither verb has any business touching the ungrouped slot.
    real = target != NO_GROUP
    if act[1].button("🗑️", key="review_group_delete", width="stretch",
                     disabled=not real,
                     help="Delete the selected group. Its columns fall back to "
                          "Uncategorized Features; no column is lost."):
        _delete_group(target)
    if act[2].button("✏️", key="review_group_rename", width="stretch",
                     disabled=not real,
                     help="Rename the selected group to the name in the box above, "
                          "carrying its columns over. A name already in use merges the "
                          "two, which is how a whole group moves in one action."):
        _rename_group(target, new_name)
    if act[3].button(f"Apply to {len(picked)}", key="review_apply_group",
                     type="primary", width="stretch", disabled=not picked,
                     help="Put the ticked rows in the selected group"):
        _apply_group(picked, target)
    # One button rather than an All/Clear pair, and not the select-all the table header
    # would conventionally carry: that header renders *inside* the scroll box, so it
    # leaves the screen on exactly the wide files a select-all is for.
    everything = len(picked) == len(numeric)
    if act[4].button("Clear" if everything else "Select all", key="review_select_all",
                     width="stretch", help="Tick every measurement row, or none"):
        if everything:
            _clear_picks(numeric)
        else:
            for col in numeric:
                st.session_state[_pick_key(col)] = True
        st.rerun()


def _apply_group(columns, group):
    """Put every ticked column in one existing group, or take them out of theirs.

    Assignment only -- `_add_group` is what makes a group. Folding the two together made
    Apply create the thing it assigned to, which is the wrong way round: the group has to
    exist before there is anywhere to assign.

    The one rule here that spans rows and fires *before* the edit comes back, against the
    grain of everything else in this file: it reads the roles and groups the last run
    enforced, writes, and reruns. Safe because that rerun re-renders the rows and
    `enforce_role_invariants` runs over them as usual -- a column ticked and then demoted
    keeps its tick for one run, and the strip is what catches it.
    """
    if not columns:
        return
    groups = dict(st.session_state._review_groups)
    for col in columns:
        if group == NO_GROUP:
            groups.pop(col, None)
        else:
            groups[col] = group
    st.session_state._review_groups = groups
    # Unticked because the action is finished. Leaving them is the worse hazard: the box
    # scrolls, so ticking five more rows on top of a forgotten twenty would quietly put
    # all twenty-five in the next group.
    _clear_picks(columns)
    _bump_editor()
    # The *destination* does stay, unlike the ticks: filling a group in two passes is
    # ordinary, and the bump would otherwise send it back to Uncategorized every time. Same
    # seeding as `_add_group`, and legal for the same reason -- the key is new after the
    # bump. Nothing is hidden by it: the dropdown sits next to the button that reads it.
    st.session_state[f"review_group_target_{st.session_state[_GEN]}"] = group
    st.rerun()


def _group_name_error(name, allow_existing=False):
    """Why this cannot name a group, or "".

    One rule for both writers, because they are one screen apart and had drifted: Rename
    accepted NO_GROUP, which put a second "—" in every row's Group dropdown. A duplicated
    option resolves through `index()` to the first slot -- the ungrouped one -- so every
    column in the renamed group silently left it, and the phantom option outlived the
    working copy.

    `allow_existing` is Rename's alone. A name already in use means *merge* there, which
    is a whole group moved in one action; at the bar it can only be a typo, since the
    dropdown beside the box already offers every live group.
    """
    # Both spellings of the ungrouped slot. A real group called "Uncategorized" would sit
    # in a measurement's dropdown reading exactly like the slot above it -- distinct
    # values, indistinguishable options, and a silent mis-pick.
    if name in (NO_GROUP, UNGROUPED_LABEL):
        return f"{code_span(name)} is how the table shows a column with no group."
    if not allow_existing and name in st.session_state._review_group_names:
        return f"There is already a group called {code_span(name)}."
    return ""


def _add_group(name):
    """Make an empty group and point the destination at it.

    The seeding is what makes "create, then assign" one flow instead of two lookups: the
    bump gives the destination selectbox a key it has never had, so writing that key is a
    plain default and not a forbidden write to a rendered widget.
    """
    name = (name or "").strip()
    if not name:
        st.warning("Type a name for the group first.")
        return
    error = _group_name_error(name)
    if error:
        st.warning(error)
        return
    st.session_state._review_group_names.append(name)
    _bump_editor()
    st.session_state[f"review_group_target_{st.session_state[_GEN]}"] = name
    st.rerun()


def _rename_group(old, new):
    """Rename a group -- or, if that name is already in use, merge into it.

    The usual correction is that the grouping is right and the name is wrong. Merging is
    the other half: renaming `lifetime` onto `morphology` moves every one of its columns
    there in a single action, where the bar would cost a tick per column.

    De-duplicated rather than rewritten in place, because a repeated entry in the name
    list is exactly the `index()` trap `_group_name_error` exists to close.
    """
    new = (new or "").strip()
    if not new:
        st.warning("Type the new name beside the group first.")
        return
    error = _group_name_error(new, allow_existing=True)
    if error:
        st.warning(error)
        return
    st.session_state._review_group_names = list(dict.fromkeys(
        new if name == old else name for name in st.session_state._review_group_names))
    st.session_state._review_groups = {
        col: (new if group == old else group)
        for col, group in st.session_state._review_groups.items()}
    _bump_editor()
    # Follow the group to its new name, as Add and Apply do. Without this the bump sent
    # the selection back to Uncategorized, greying out Delete and Rename on the group the
    # user had just been working on. Nothing to seed after a *delete*: that name is gone,
    # and the prune in `_group_section` puts the selection back on Uncategorized itself.
    st.session_state[f"review_group_target_{st.session_state[_GEN]}"] = new
    st.rerun()


def _delete_group(name):
    st.session_state._review_group_names = [
        group for group in st.session_state._review_group_names if group != name]
    st.session_state._review_groups = {
        col: group for col, group in st.session_state._review_groups.items()
        if group != name}
    _bump_editor()
    st.rerun()


def exit_actions(applied, reopened):
    """The buttons the last row offers, in order, as `(kind, profile)` pairs.

    Exactly one of them writes, always -- there is no "use without saving". Every file
    that reaches a plot is therefore described by a profile on disk, which is what lets
    the *file* pick the profile on the next upload rather than the user.

    Which write it is follows the profile in force: a working copy that came from a
    profile goes back to that profile, and only a copy that came from nowhere may name
    a new one. That is the whole of the rule that keeps two profiles from ever holding
    the same column set -- a profile can only acquire a column set at a moment when no
    profile had it.

    `cancel` exists only on a reopening, because only then is there a previous decision
    to fall back to. It has to exist there: without it the sole way out of a table
    opened out of curiosity would be a write to the profile.
    """
    actions = [("save", applied) if applied else ("save_as_new", None)]
    if reopened:
        actions.append(("cancel", None))
    return actions


def _buttons(df, saved_names):
    roles = st.session_state._review_roles
    groups = st.session_state._review_groups
    names = st.session_state._review_group_names
    source = _applied_profile()
    blocked = review_blocking_reason(df, roles)
    actions = exit_actions(source, st.session_state.get("_review_reopened", False))

    # Save as takes two slots, because its name box sits beside it rather than behind it.
    widths = [width for kind, _ in actions
              for width in ((1, 2) if kind == "save_as_new" else (1,))]
    cols = st.columns([*widths, 2])
    slots = iter(cols)
    for kind, profile in actions:
        if kind == "save":
            with next(slots):
                if st.button(f"💾 Save to {profile} & use", type="primary",
                             width="stretch", disabled=bool(blocked)):
                    _save_and_close(profile, roles, groups, names)
        elif kind == "save_as_new":
            _save_as_new(next(slots), next(slots), roles, groups, names, blocked,
                         saved_names)
        else:
            with next(slots):
                if st.button("Cancel", width="stretch",
                             help="Discard these edits and go back"):
                    # Reload from disk rather than merely closing: the working copy holds
                    # the edits, and confirming it would use them without ever saving.
                    _load_working_copy(df, source)
                    st.session_state._review_confirmed = True
                    st.rerun()
    # Only the blocked reason, and only when there is one. What saving does is already
    # written on the button that does it.
    if blocked:
        cols[-1].error(blocked)


def _save_as_new(button_slot, name_slot, roles, groups, names, blocked, saved_names):
    """The button and, to its right, the box naming what it writes.

    Every new file shape has to make this write -- nothing reaches a plot unsaved -- so
    the box is on screen and typed into directly rather than waiting behind a popover for
    a click that only reveals it.

    The box is *rendered* first and placed second: the button's `if` needs the name in
    hand on the run the click arrives, and slots are containers, so code order and screen
    order are free to disagree.

    No form around the pair, so Return does not save: it reruns, and the typed name
    survives in session state, which is the whole cost. A form would have to wrap both
    columns, and this row is already a column inside the page's own.

    The placeholder instructs rather than illustrates. A profile is a *column set*, so
    every replicate of an experiment matches the same one -- an example like `pdl1-rep3`
    teaches exactly the wrong instinct, and the name it suggests would be a lie by the
    second file.

    The count in the tooltip comes off `saved_names`, the caller's one read of the config
    -- see `_render_gate`. `save_working_copy` re-reads it at the moment of the write,
    which is the read that has to be fresh.
    """
    with name_slot:
        new_name = st.text_input(
            "Profile name", key=f"review_save_as_name_{_file_gen()}",
            label_visibility="collapsed", placeholder="name this profile",
            help=f"{len(saved_names)} of {MAX_PROFILES} profiles saved. "
                 "An existing name overwrites that profile.")
    with button_slot:
        if st.button("💾 Save profile as", type="primary", width="stretch",
                     disabled=bool(blocked)):
            _save_and_close(new_name, roles, groups, names)


def _save_and_close(name, roles, groups, group_names):
    """Write the working copy, then leave. Saving *is* the way out.

    One button, not a save and a separate "use": splitting them lets one open table write
    two profiles over the same columns, and leaves the user unsure whether a green message
    means the table is done with.
    """
    error = save_working_copy(name, roles, groups, group_names=group_names)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    st.session_state._review_saved_as = name.strip()
    st.session_state._review_confirmed = True
    st.toast(f"Saved to {code_span(name.strip())} {happy_emoji}")
    st.rerun()


def reopen_gate():
    """Put the review table back over a file that is already using a profile.

    The ✏️ beside the summary, and the only way back in: an exact match applies without
    a click, so a file can reach the plots -- or be rejected -- having never shown its
    owner a table, and re-uploading it is no escape, because the fingerprint is the name
    and the columns, both unchanged. Auto-apply does not re-fire (`_review_opened`
    survives), so this reopens on the roles in force rather than looping on the match
    that produced them.

    Two things are recorded, both read once and then fixed for this opening. `reopened`
    is what puts a Cancel in the button row -- there is a previous decision to fall back
    to, which is not true of a first open. And the chooser is asked for only if the
    profile in force does not already describe this file exactly; when it does there is
    nothing to choose, and offering the others would invite writing this file's column
    set onto a second profile.
    """
    st.session_state._review_confirmed = False
    # Auto-apply returns *before* the line that sets `_review_opened`, so on that path
    # the flag was never written and clearing `_review_confirmed` alone let the match
    # fire again on the very next run -- the table reopened and shut in the same breath.
    st.session_state._review_opened = True
    st.session_state._review_reopened = True
    st.session_state._review_chooser = chooser_is_needed(
        _applied_profile(), set(st.session_state.get("_review_roles") or {}),
        all_profile_columns())
