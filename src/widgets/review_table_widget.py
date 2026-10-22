"""Column review UI for editing a session-local working copy.

Profile matching and role rules live in ``profile_matching`` and ``column_roles``.
Review hides analysis until Use, Save, or Cancel resumes it. Edited roles and groups
reach the saved profile only through ``save_working_copy``.
"""
import html
from copy import deepcopy

import streamlit as st

from src.analysis_config import (
    AUTO_DETECT,
    MANAGE_LABEL,
    MAX_PROFILES,
    all_profile_columns,
    delete_profile,
    get_analysis_hints,
    profile_roles_and_groups,
    rename_profile,
    save_working_copy,
)
from src.column_roles import (
    LABEL_ROLES,
    NO_GROUP,
    ROLE_CATEGORICAL,
    ROLE_IGNORE,
    ROLE_LABELS,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    UNGROUPED_LABEL,
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
from src.natural_sort import natural_key
from src.profile_matching import (
    exact_match,
    rank_profiles,
)
from src.widgets.analysis_widget_state import (
    analysis_control_keys,
    preserve_analysis_controls,
)

ROLE_HELP = ("Use each column's **Role** dropdown to choose how it is used:\n\n"
             "- **Row ID**: one column of unique identifiers with no missing values. "
             "Optional; row numbers are generated if none is assigned.\n"
             "- **Categorical**: labels for grouping or filtering, such as treatment, "
             "donor, or batch, even when encoded as numbers.\n"
             "- **Numerical**: measurements to plot or analyze, such as lifetime "
             "or cell area.\n"
             "- **Ignore**: exclude the column from analysis.\n\n"
             "Only Numerical columns can be assigned to feature groups.")
GROUP_HELP = ("Each group becomes one dropdown in the feature pickers, so features are "
              "chosen from a few short lists instead of one long one. Measurements left "
              "ungrouped are collected under Uncategorized Features.")

# Clear the working copy and review state together when the file fingerprint changes.
_STATE_KEYS = (
    "_review_fingerprint", "_review_roles", "_review_groups", "_review_group_names",
    "_review_source", "_review_confirmed", "_review_previous_roles", "_review_known_cols",
    "_review_notices", "_review_opened", "_review_saved_as", "_review_numeric_cols",
    "_review_delete_armed", "_review_manage_open",
    "_review_configured_row_id", "_review_picks_stale", "_review_overwrite_armed",
    "_review_preserve_controls", "_review_snapshot", "_review_baseline",
    "_review_toast", "_review_bulk_pending",
)
# Export needs the configured Row ID, including blank for generated row numbers.
# Keep this key in the same file-scoped reset as the rest of review state.
_CONFIGURED_ROW_ID = "_review_configured_row_id"
# The profile awaiting deletion confirmation; only one row can be armed.
_DELETE_ARMED = "_review_delete_armed"
# Keep management open after its controls trigger a remount.
_MANAGE_OPEN = "_review_manage_open"
# The profile awaiting overwrite confirmation; confirmation applies only while
# the trimmed input matches this name and the profile still exists.
_OVERWRITE_ARMED = "_review_overwrite_armed"
# Read this button key before rendering the gate; its button appears in the summary.
_REOPEN = "review_reopen"
# Roles the bar's role row offers. Row ID is left out: only one column can hold it.
_BULK_ROLES = (ROLE_CATEGORICAL, ROLE_NUMERICAL, ROLE_IGNORE)
# Reassigned before every render, so its tick always reflects the row ticks.
_SELECT_ALL = "review_select_all"
# Buttons above the rows that end their run with st.rerun().
_BAR_BUTTONS = ("review_apply_role", "review_apply_group", "review_add_group",
                "review_group_rename", "review_group_delete")
# Row label, dropdown, button, New group, ✏️. Both rows use all five slots, because
# Streamlit spreads gaps by column count: rows of 6 and 5 columns drift 2px apart.
# Measured in the ~900px editor column of a 1440px window: every slot stays on one
# line, so the bar keeps its height while the button labels change.
_BAR_WEIGHTS = (4.8, 2.7, 2.8, 2.3, 1.2)
# Streamlit gives a column's first checkbox an 8px top margin, which sits it 4px below
# the centre of a vertically centred row. A collapsed label shrinks to the box and its
# 4px top margin; padding restores the text line height that centres the box. The
# table's border and padding total 1rem, so the same indent puts Select all over the ticks.
_TICK_STYLE = ("<style>.st-key-review_select_all{margin-left:1rem}"
               ".st-key-review_select_all .stCheckbox,"
               "[class*='st-key-review_tick_'] .stCheckbox{margin-top:0 !important}"
               "[class*='st-key-review_tick_'] label{padding-bottom:0.25rem}</style>")
# Selection, new badge, name, role, group, preview. The group slot is wider
# than the role slot to fit UNGROUPED_LABEL.
_ROW_WEIGHTS = (1, 1, 5, 3, 4, 4)
# Include widget gaps and padding; cap tall tables so Save stays near the rows.
_ROW_HEIGHT, _HEADER_HEIGHT, _MAX_TABLE_HEIGHT = 56, 78, 470

_GEN = "_review_editor_gen"
# File-scoped keys preserve selections and typed names through editor corrections
# while resetting them when the file fingerprint changes.
_FILE_GEN = "_review_file_gen"


_COPY_KEYS = (
    "_review_roles", "_review_groups", "_review_group_names", "_review_source",
    "_review_saved_as", "_review_known_cols", "_review_numeric_cols", "_review_baseline",
    "_review_configured_row_id",
)


def reset_upload_review():
    """Uploader on_change callback: discard decisions even for identical file names."""
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state[_FILE_GEN] = _file_gen() + 1
    _bump_editor()


def _assignments():
    return deepcopy((st.session_state.get("_review_roles", {}),
                     st.session_state.get("_review_groups", {}),
                     st.session_state.get("_review_group_names", [])))


def _status():
    changed = _assignments() != st.session_state.get("_review_baseline")
    name = _applied_profile()
    if name:
        return f"{name} · edited for this upload" if changed else name
    return "This upload only" if changed else "Auto-detected"


def _fingerprint(uploaded_file, df):
    """Identify review state by upload name and ordered column names."""
    return (getattr(uploaded_file, "name", ""), tuple(df.columns))


def _file_gen():
    return st.session_state.get(_FILE_GEN, 0)


def _table_height(columns):
    """Tall enough for the rows, capped at a box the Save button still fits under."""
    return min(_MAX_TABLE_HEIGHT, _HEADER_HEIGHT + _ROW_HEIGHT * columns)


def _bump_editor():
    """Re-key rows to apply programmatic corrections on the next rerun.

    Rendered widget keys cannot be assigned, and deleting them can restore
    stale browser values.
    """
    st.session_state[_GEN] = st.session_state.get(_GEN, 0) + 1


def _group_label(name):
    """Display the measurement destination for the shared ``NO_GROUP`` option.

    A ``format_func`` keeps every row's stored options identical.
    """
    return UNGROUPED_LABEL if name == NO_GROUP else name


def _group_key(gen, col, measurement):
    """Include label mode in a Group selectbox's identity.

    Re-keying refreshes the ungrouped label after a role change; changing
    ``format_func`` alone can leave the browser's option labels stale.
    """
    return f"review_group_{gen}_{'num' if measurement else 'other'}_{col}"


def _pick_key(col):
    """Keep row selections across editor corrections and reset them for a new file."""
    # Use a distinct prefix from chooser buttons to avoid profile/column-name collisions.
    return f"review_tick_{_file_gen()}_{col}"


def _picked_columns(df, numerical_only=False):
    """Return selected rows in file order, or only the selected measurements.

    Read before rendering rows to include current clicks. Group actions pass
    numerical_only because only a Numerical column can sit in a group. The Row ID
    is never picked: a browser keeps the tick it had before its lock replaced the
    checkbox.
    """
    roles = st.session_state._review_roles
    return [col for col in df.columns
            if st.session_state.get(_pick_key(col), False)
            and roles.get(col) != ROLE_ROW_ID
            and (not numerical_only or roles.get(col) == ROLE_NUMERICAL)]


def _clear_picks(columns):
    """Clear selections before their widgets render.

    Assignment updates widget state; deleting a key can restore the browser's
    previous value.
    """
    for col in columns:
        st.session_state[_pick_key(col)] = False


def _load_working_copy(df, picked):
    """Build a new draft from saved decisions and read-only hints, replacing edits."""
    source = None if picked == AUTO_DETECT else picked
    profile_roles, profile_groups, profile_names = ({}, {}, [])
    if source:
        profile_roles, profile_groups, profile_names = profile_roles_and_groups(source)
    roles, groups, numeric_cols = build_working_copy(
        df, profile_roles, profile_groups, profile_group_names=profile_names,
        **get_analysis_hints())
    st.session_state._review_roles = roles
    st.session_state._review_groups = groups
    # Preserve stored group order and empty groups before adding newly detected groups.
    # The reserved ungrouped label cannot also name a real group.
    names = [name for name in dict.fromkeys(profile_names) if name not in (NO_GROUP, UNGROUPED_LABEL)]
    for col in df.columns:
        group = groups.get(col)
        if group and group not in names:
            names.append(group)
    st.session_state._review_group_names = names
    st.session_state._review_source = picked
    # A saved source baseline includes absent columns, so schema changes count as edits.
    baseline_groups = {col: group for col, group in profile_groups.items()
                       if profile_roles.get(col) == ROLE_NUMERICAL and group
                       and group not in (NO_GROUP, UNGROUPED_LABEL)}
    baseline_names = [name for name in dict.fromkeys(profile_names)
                      if name not in (NO_GROUP, UNGROUPED_LABEL)]
    st.session_state._review_baseline = deepcopy(
        (profile_roles, baseline_groups, baseline_names) if source else (roles, groups, names))
    # A previously saved name no longer identifies this replacement working copy.
    st.session_state.pop("_review_saved_as", None)
    st.session_state._review_previous_roles = dict(roles)
    st.session_state._review_known_cols = set(profile_roles)
    # Reuse the numeric-column set from role detection; coercing again is expensive.
    st.session_state._review_numeric_cols = numeric_cols
    # Cancel calls this below rendered rows. Defer checkbox writes until the next
    # _bulk_bar call, before those widgets render.
    st.session_state._review_picks_stale = True
    _bump_editor()


def _applied_profile():
    """Return the working copy's saved or selected profile.

    A successful Save name takes precedence over the chooser selection.
    ``current_profile`` may instead name a different file's last write.
    """
    source = (st.session_state.get("_review_saved_as")
              or st.session_state.get("_review_source"))
    return None if source in (None, AUTO_DETECT) else source


def applied_summary(decision):
    """Show the applied profile or auto-detection label and assigned role counts.

    Count the full role decision, including columns removed during loading.
    """
    roles = decision.get("roles") or {}
    if not roles:
        return
    counts = dict.fromkeys(ROLE_LABELS, 0)
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1
    # Omit the optional single Row ID from the role-count summary.
    tally = " · ".join(f"{n} {ROLE_LABELS[role]}" for role, n in counts.items()
                       if n and role != ROLE_ROW_ID)
    name = _status()
    # The summary provides review access even when an exact match skips the gate.
    # Content-width children and nowrap keep the pencil beside the summary text.
    with st.container(key="review_summary", horizontal=True,
                      vertical_alignment="center", gap=None):
        # Inline the style to avoid an extra layout gap and keep this line with the upload.
        # Escape the user-provided profile name at its HTML interpolation point.
        st.markdown(
            # Keep the pencil on the same line without shrinking its click target.
            "<style>.st-key-review_summary{margin-top:-0.5rem;flex-wrap:nowrap}"
            # Reset wrapper margins and align each wrapper to the text line height.
            ".st-key-review_summary>div:first-child,.st-key-review_summary .stMarkdown,"
            ".st-key-review_summary .stMarkdown div"
            "{height:1.5rem;display:flex;align-items:center;margin-bottom:0}"
            ".st-key-review_reopen{flex:0 0 auto}"
            ".st-key-review_reopen button{min-height:0;height:1.5rem;padding:0 0.4rem}"
            ".st-key-review_reopen p{line-height:1.5rem}</style>"
            "<div style='font-size:0.875rem;opacity:0.65;margin:0;line-height:1.5rem;"
            f"white-space:nowrap'><b>{html.escape(name)}</b>:&nbsp;{tally}</div>",
            unsafe_allow_html=True, width="content")
        # Handle the click in review_gate before its slot renders, so review opens
        # and analysis settings are preserved in the same run.
        st.button("✏️", key=_REOPEN, type="tertiary",
                  help="Review this file's columns again")


def _decision():
    roles = st.session_state.get("_review_roles", {})
    # Capture the configured identifier before loading can generate a row-number column.
    # Use the same first Row ID as apply_column_roles.
    st.session_state[_CONFIGURED_ROW_ID] = next(
        (col for col, role in roles.items() if role == ROLE_ROW_ID), "")
    return {
        "profile": _applied_profile(),
        "roles": roles,
        "groups": st.session_state.get("_review_groups", {}),
        "group_names": st.session_state.get("_review_group_names", []),
    }


def configured_row_id():
    """Return the configured Row ID column, or "" for generated row numbers.

    Export needs the configured name so it can create row numbers itself,
    without requiring the app's generated column in the input file.
    """
    return st.session_state.get(_CONFIGURED_ROW_ID, "")


def ignored_columns():
    """Return ignored columns for export's numeric-coercion skip set."""
    roles = st.session_state.get("_review_roles", {})
    return [col for col, role in roles.items() if role == ROLE_IGNORE]


def review_gate(uploaded_file, df):
    """Return the confirmed decision, or render review and return None.

    Initial entry validates a unique exact profile or inference immediately.
    While review is open, analysis is hidden and captured controls are preserved.
    Load profile lists as needed and share them with child renderers.
    """
    profiles = None
    # Widget state arrives before the script body, so reopening needs no extra rerun.
    # The confirmed guard prevents the click from reopening an already open gate.
    if st.session_state.get("_review_confirmed") and st.session_state.get(_REOPEN):
        reopen_gate()
    fingerprint = _fingerprint(uploaded_file, df)
    if st.session_state.get("_review_fingerprint") != fingerprint:
        reset_upload_review()
        st.session_state._review_fingerprint = fingerprint

    if not st.session_state.get("_review_confirmed") and not st.session_state.get("_review_opened"):
        profiles = all_profile_columns()
        matched = exact_match(set(df.columns), profiles)
        _load_working_copy(df, matched or AUTO_DETECT)
        st.session_state._review_confirmed = not review_blocking_reason(
            df, st.session_state._review_roles)

    control_keys = st.session_state.get("_review_preserve_controls")
    if control_keys:
        preserve_analysis_controls(st.session_state, control_keys)

    if st.session_state.get("_review_confirmed"):
        return _decision()

    st.session_state._review_opened = True
    _render_gate(uploaded_file, df, profiles)
    # Review hides analysis, including on reopen. Use, Save, and Cancel rerun before
    # a confirmed decision reaches the page.
    return None


def _render_gate(uploaded_file, df, profiles=None):
    """Render optional saved profiles, the populated editor, exits, and management."""
    name = getattr(uploaded_file, "name", "the uploaded file")
    # Render the filename literally inside the Markdown caption.
    st.caption(f"Read {len(df.columns)} columns × {len(df):,} rows from {code_span(name)}")

    if profiles is None:
        profiles = all_profile_columns()
    saved_names = list(profiles)
    _chooser(df, profiles)

    for notice in st.session_state.pop("_review_notices", []):
        st.info(notice)
    # A toast reports the last bulk change without taking page height.
    show_review_toast()
    _bulk_bar(df)
    _editor(df)
    # Use the corrected roles to announce row numbers before they are generated.
    numbering = row_id_notice(st.session_state._review_roles)
    if numbering:
        st.info(numbering)
    _buttons(df, saved_names)
    _manage_profiles(saved_names, df)


def _chooser(df, profiles):
    """Offer related saved profiles in rank order on every editor opening."""
    fits = [fit for fit in rank_profiles(set(df.columns), profiles) if fit.shared]
    if not fits:
        return
    st.markdown("**Apply a saved profile (optional)**")
    picked = st.session_state.get("_review_source")
    for fit in fits:
        _pick_row(df, fit.name, f"{fit.name}  —  {len(fit.shared)} shared · "
                  f"{len(fit.missing)} missing · {len(fit.new)} new", picked)
    st.caption("Applying a profile replaces the draft below and keeps review open.")


def _pick_row(df, name, label, picked):
    """Load a clicked chooser option and rerun to update every row's selection."""
    if st.button(label, key=f"review_pick_{name}",
                 type="secondary" if name == picked else "tertiary"):
        _load_working_copy(df, name)
        st.session_state._review_confirmed = False
        st.rerun()


def _manage_profiles(saved_names, df=None):
    """Render rename/delete controls for all saved profiles, sorted by name.

    Keep the panel open after actions that remount it. Use a stable panel label
    and keys tied to profile names to preserve widget identity through deletions.
    New files reset the panel state.
    """
    names = sorted(saved_names, key=natural_key)
    if not names:
        return
    with st.expander(MANAGE_LABEL, expanded=st.session_state.get(_MANAGE_OPEN, False)):
        for name in names:
            with st.container(key=f"review_manage_row_{name}"):
                if st.session_state.get(_DELETE_ARMED) == name:
                    _delete_confirm(name, df)
                else:
                    _manage_row(name)


def _manage_row(name):
    cols = st.columns([12, 1, 1], vertical_alignment="center")
    with cols[0]:
        # Profile names are plain text, including any Markdown characters.
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
    """Use a form so Return submits the rename before closing the popover."""
    with st.popover("✏️", help=f"Rename {name}", width="stretch"), st.form(
            f"review_rename_form_{name}", border=False):
        # Reset unsubmitted names for a new file.
        new_name = st.text_input("Rename to", value=name,
                                 key=f"review_rename_{_file_gen()}_{name}")
        if st.form_submit_button("Rename", key=f"review_rename_submit_{name}"):
            _rename_and_refresh(name, new_name)


def _delete_confirm(name, df=None):
    """Confirm deletion in place for the profile named by ``_DELETE_ARMED``.

    The armed profile name is server state, so confirmation cannot move to a
    neighboring row after deleting a profile.
    """
    cols = st.columns([8, 2, 2], vertical_alignment="center")
    with cols[0]:
        st.text(f"Delete {name}? Its roles and groups go with it.")
    with cols[1]:
        if st.button("Delete", key=f"review_delete_{name}", type="primary", width="stretch"):
            _delete_and_refresh(name, df)
    with cols[2]:
        if st.button("Keep", key=f"review_delete_cancel_{name}", width="stretch"):
            st.session_state.pop(_DELETE_ARMED, None)
            st.session_state[_MANAGE_OPEN] = True
            st.rerun()


def _rename_and_refresh(old, new):
    """Rename a profile and update both working-copy references to its name."""
    error = rename_profile(old, new)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    for key in ("_review_source", "_review_saved_as"):
        if st.session_state.get(key) == old:
            st.session_state[key] = (new or "").strip()
    snapshot = st.session_state.get("_review_snapshot")
    if snapshot:
        for key in ("_review_source", "_review_saved_as"):
            if snapshot.get(key) == old:
                snapshot[key] = (new or "").strip()
    st.session_state[_MANAGE_OPEN] = True
    st.rerun()


def _delete_and_refresh(name, df=None):
    """Invalidate deleted snapshot sources; rebuild a deleted draft with remaining hints."""
    error = delete_profile(name)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    # Clear the deleted profile from confirmation state before rerunning.
    st.session_state.pop(_DELETE_ARMED, None)
    # Keep management open when the changed row list remounts it.
    st.session_state[_MANAGE_OPEN] = True
    snapshot = st.session_state.get("_review_snapshot") or {}
    snapshot_source = snapshot.get("_review_saved_as") or snapshot.get("_review_source")
    if name == snapshot_source or name == _applied_profile():
        st.session_state.pop("_review_snapshot", None)
    if name == _applied_profile():
        _load_working_copy(df, AUTO_DETECT)
    st.session_state._review_confirmed = False
    st.rerun()


def _editor(df):
    """Render role/group controls in a scrollable row per file column.

    Names and previews use plain text because they come from the file.
    Validate cross-row rules after collecting edits and re-key corrected rows.
    """
    roles = st.session_state._review_roles
    groups = st.session_state._review_groups
    known = st.session_state.get("_review_known_cols") or set()
    # Use the same numeric classification for previews and role detection.
    numeric_cols = st.session_state.get("_review_numeric_cols") or set()
    role_options = list(ROLE_LABELS.values())
    group_options = [NO_GROUP] + st.session_state._review_group_names
    gen = st.session_state[_GEN]

    edited_roles, edited_groups = {}, {}
    with st.container(height=_table_height(len(df.columns)), border=True):
        header = st.columns(_ROW_WEIGHTS, vertical_alignment="center")
        for slot, title, tip in zip(header[2:], ("Column", "Role", "Group", "Preview"),
                                    (None, ROLE_HELP, GROUP_HELP, None)):
            slot.caption(f"**{title}**", help=tip)
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
                # Re-key when the role changes the ungrouped label.
                key=_group_key(gen, col, measurement),
                index=group_options.index(held) if held in group_options else 0,
                label_visibility="collapsed",
                # Both display labels use the same stored NO_GROUP value.
                format_func=_group_label if measurement else str,
                # The cross-row invariant check also clears groups after role changes.
                disabled=not measurement,
                help="Only a Numerical column can sit in a group. Ungrouped measurements "
                     "fall to Uncategorized Features.")
            # A lock stands in for the Row ID's tick; _picked_columns ignores any tick
            # it kept. A disabled button keeps the tooltip on the lock (a Markdown help
            # icon wraps here); it is keyed per column because two rows hold Row ID
            # until the correction reruns.
            if LABEL_ROLES[role] == ROLE_ROW_ID:
                cells[0].button(":material/lock:", key=f"review_lock_{col}",
                                type="tertiary", disabled=True,
                                help="Select all and bulk changes skip the Row ID. "
                                     "Change it with its Role dropdown.")
            else:
                cells[0].checkbox(f"Select {col}", key=_pick_key(col),
                                  label_visibility="collapsed",
                                  help="Tick to include in a bulk change")
            # Mark columns whose roles were guessed because the profile did not contain them.
            if known and col not in known:
                cells[1].markdown(":orange-badge[new]")
            cells[2].text(col)
            cells[5].text(column_preview(df[col], numeric=col in numeric_cols))
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
        # Re-key to display corrected values; carry their notices into the rerun.
        st.session_state._review_notices = notices
        _bump_editor()
        st.rerun()


def _bulk_key():
    """Keep the group row's pick through editor corrections; reset it for a new file."""
    return f"review_bulk_action_{_file_gen()}"


def _bulk_button_label(kind, value, n_picked, n_targets):
    """Name the change the button makes, or why it cannot make one."""
    if not n_picked:
        return "Tick rows below"
    if kind == "group":
        if not n_targets:
            return "No Numerical ticked"
        return f"Ungroup {n_targets}" if value == NO_GROUP else f"Move {n_targets}"
    if value == ROLE_IGNORE:
        return f"Ignore {n_targets}"
    return f"Make {n_targets} {ROLE_LABELS[value]}"


def _point_bar_at(group):
    """Pick a group in the group row on the next run.

    Group tools render after the row's dropdown, so they cannot assign its key.
    """
    st.session_state._review_bulk_pending = group


def _select_all(columns):
    """Tick every row except the Row ID, or clear every row.

    Read roles here rather than from the bar's last render: a row made the Row ID
    since then would otherwise be ticked.
    """
    if st.session_state[_SELECT_ALL]:
        roles = st.session_state._review_roles
        for col in columns:
            if roles.get(col) != ROLE_ROW_ID:
                st.session_state[_pick_key(col)] = True
    else:
        _clear_picks(columns)


def _bulk_bar(df):
    """Render the role row and the group row that change the ticked rows.

    This bar must run before the row checkboxes because it writes their
    session-state values. Each row's dropdown keeps its last choice.
    """
    # Apply deferred resets before the row checkboxes render.
    if st.session_state.pop("_review_picks_stale", False):
        _clear_picks(df.columns)
    roles = st.session_state._review_roles
    options = [NO_GROUP, *st.session_state._review_group_names]
    key = _bulk_key()
    pending = st.session_state.pop("_review_bulk_pending", None)
    held = pending if pending in options else st.session_state.get(key, NO_GROUP)
    # A deleted group falls back to Uncategorized, never to the next group, so the
    # ✏️ box only acts on a group someone picked.
    st.session_state[key] = held if held in options else NO_GROUP

    picked = _picked_columns(df)
    # st.rerun() counts as a finished run, so a button here would drop the rows'
    # unrendered ticks. Reassert them in that run only, so other runs never resend
    # tick values over a click still in flight; the change then clears its own.
    if any(st.session_state.get(key) for key in _BAR_BUTTONS):
        for col in picked:
            st.session_state[_pick_key(col)] = True
    # Leave the Row ID out so one bulk change cannot remove the identifier by accident.
    eligible = [col for col in df.columns if roles.get(col) != ROLE_ROW_ID]
    st.session_state[_SELECT_ALL] = bool(eligible) and set(eligible) <= set(picked)

    # Style-only st.html goes to the event container, so it adds no block gap.
    st.html(_TICK_STYLE)
    # Right-aligned labels make each row read as a sentence ending at its dropdown.
    cells = st.columns(_BAR_WEIGHTS, vertical_alignment="center")
    # Select all shares the label slot, so both rows keep the same columns.
    with cells[0].container(horizontal=True, horizontal_alignment="distribute",
                            vertical_alignment="center", gap=None):
        st.checkbox("Select all", key=_SELECT_ALL, on_change=_select_all,
                    args=(list(df.columns),), disabled=not eligible,
                    help="Tick every row except the Row ID, or clear every row")
        st.markdown("Ticked columns are", help=ROLE_HELP, width="content")
    role = cells[1].selectbox(
        "Role for ticked rows", _BULK_ROLES, index=_BULK_ROLES.index(ROLE_IGNORE),
        key=f"review_bulk_role_{_file_gen()}", format_func=ROLE_LABELS.get,
        label_visibility="collapsed")
    if cells[2].button(_bulk_button_label("role", role, len(picked), len(picked)),
                       key="review_apply_role", type="primary", width="stretch",
                       disabled=not picked, help="Give the ticked rows the role on the left"):
        _apply_role(picked, role)

    measurements = _picked_columns(df, numerical_only=True)
    skipped = [col for col in picked if col not in measurements]
    numeric = ROLE_NUMERICAL in roles.values()
    cells = st.columns(_BAR_WEIGHTS, vertical_alignment="center")
    cells[0].markdown("Ticked **Numerical** go in group", help=GROUP_HELP,
                      text_alignment="right")
    # Omit index: the session value seeded above selects the option.
    group = cells[1].selectbox(
        "Group for ticked Numerical rows", options, key=key, format_func=_group_label,
        label_visibility="collapsed", disabled=not numeric)
    if cells[2].button(_bulk_button_label("group", group, len(picked), len(measurements)),
                       key="review_apply_group", type="primary", width="stretch",
                       disabled=not measurements,
                       help="Put the ticked Numerical rows in the group on the left"):
        _apply_group(measurements, group, skipped)
    _group_section(cells[3:], group, measurements, skipped, numeric)


def _group_section(cells, target, members, skipped, numeric):
    """Render ➕ New group and the ✏️ box at the end of the group row.

    Each names a group in a popover form, so Return submits its first button. New
    group also moves the ticked measurements in; the ✏️ box renames or deletes
    `target`, the group picked in the row's dropdown. Uncategorized is no group, so
    the box stays shut on it.
    """
    gen = st.session_state[_GEN]
    with cells[0].popover("➕ New group", width="stretch", disabled=not numeric,
                          help="Make a group holding the ticked Numerical rows"), st.form(
            "review_group_new_form", border=False):
        name = st.text_input("Name of the new group", key=f"review_group_name_{gen}")
        if st.form_submit_button(f"Create with {len(members)}" if members else "Create",
                                 key="review_add_group", type="primary"):
            _add_group(name, members, skipped)
    shut = target == NO_GROUP or not numeric
    with cells[1].popover("✏️", width="stretch", disabled=shut,
                          help="Rename or delete the group picked on the left. A name "
                               "already in use merges the two groups."), st.form(
            "review_group_edit_form", border=False):
        # No help icon: placed before the box, it would take the popover's opening focus.
        new_name = st.text_input("Rename to", value="" if target == NO_GROUP else target,
                                 key=f"review_group_rename_to_{gen}_{target}")
        # Rename comes first because Return submits a form's first button. Stacked: the
        # popover is too narrow for both labels on one line. AppTest finds buttons inside
        # a disabled popover, so they repeat its disabled state.
        if st.form_submit_button("Rename", key="review_group_rename", type="primary",
                                 width="stretch", disabled=shut):
            _rename_group(target, new_name)
        if st.form_submit_button("🗑️ Delete group", key="review_group_delete",
                                 width="stretch", disabled=shut,
                                 help="Its columns go to Uncategorized; no column is lost."):
            _delete_group(target)


def _count(columns):
    return "1 column" if len(columns) == 1 else f"{len(columns)} columns"


def _named(columns, limit=3):
    """Name up to `limit` columns for a status message, then count the rest."""
    shown = ", ".join(code_span(col) for col in columns[:limit])
    rest = len(columns) - limit
    return f"{shown} and {rest} more" if rest > 0 else shown


def _apply_role(columns, role):
    """Give the selected columns one role and report the change in a toast.

    Drop the groups of columns leaving Numerical here: left to the next editor
    render, the correction's rerun would unmount the toast before it is seen.
    """
    # A click on a button label one run stale can arrive with nothing ticked.
    if not columns:
        return
    groups = st.session_state._review_groups
    lost = [] if role == ROLE_NUMERICAL else [col for col in columns if col in groups]
    st.session_state._review_roles = {**st.session_state._review_roles,
                                      **dict.fromkeys(columns, role)}
    st.session_state._review_groups = {col: group for col, group in groups.items()
                                       if col not in lost}
    verb = "Ignored" if role == ROLE_IGNORE else f"Made {ROLE_LABELS[role]}"
    message = f"{verb} {_count(columns)}: {_named(columns)}."
    if lost:
        message += (" 1 measurement left its group." if len(lost) == 1
                    else f" {len(lost)} measurements left their group.")
    st.session_state._review_toast = message
    _clear_picks(columns)
    _bump_editor()
    st.rerun()


def _apply_group(columns, group, skipped=()):
    """Assign selected columns to an existing group, or remove their assignments.

    The next render enforces role invariants, including removal of groups from
    columns that have become non-numerical. The toast names the ticked `skipped`
    rows, which keep their ticks, so they do not look ignored.
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
    toast = (f"Ungrouped {_count(columns)}: {_named(columns)}." if group == NO_GROUP else
             f"Moved {_count(columns)} to {code_span(group)}: {_named(columns)}.")
    if skipped:
        # Lead with the skip: a toast past 104 characters hides the rest behind "view more".
        toast = f"Only Numerical columns can be grouped: skipped {_named(skipped)}. {toast}"
    st.session_state._review_toast = toast
    # Clear completed selections so offscreen rows are not included in the next action.
    # Ticked rows the group could not take keep their ticks for their next change.
    _clear_picks(columns)
    _bump_editor()
    st.rerun()


def _group_name_error(name, allow_existing=False):
    """Validate Add/Rename names, allowing existing names only for merges."""
    # Reserve both ungrouped spellings to avoid indistinguishable dropdown options.
    if name in (NO_GROUP, UNGROUPED_LABEL):
        return f"{code_span(name)} is how the table shows a column with no group."
    if not allow_existing and name in st.session_state._review_group_names:
        return f"There is already a group called {code_span(name)}."
    return ""


def _add_group(name, members=(), skipped=()):
    """Create a group, pick it in the group row, and move `members` into it.

    `_apply_group` reports the move, leading with the ticked rows it `skipped`.
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
    _point_bar_at(name)
    if members:
        _apply_group(members, name, skipped)
    _bump_editor()
    st.rerun()


def _rename_group(old, new):
    """Rename a group, merging its columns if the destination already exists."""
    new = (new or "").strip()
    if not new:
        st.warning("Type the new name first.")
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
    # The ✏️ box renames the group row's pick, so the pick follows the new name.
    _point_bar_at(new)
    st.rerun()


def _delete_group(name):
    """Delete a group; its columns fall back to Uncategorized, and the toast counts
    them because some may sit below the table's scroll area.
    """
    members = [col for col, group in st.session_state._review_groups.items()
               if group == name]
    st.session_state._review_group_names = [
        group for group in st.session_state._review_group_names if group != name]
    st.session_state._review_groups = {
        col: group for col, group in st.session_state._review_groups.items()
        if group != name}
    st.session_state._review_toast = f"Deleted group {code_span(name)}" + (
        f"; {_count(members)} moved to Uncategorized." if members else ".")
    _bump_editor()
    st.rerun()


def exit_actions(applied, reopened):
    """Offer upload-only Use, optional persistence, and Cancel for a valid snapshot."""
    actions = [("use", None), ("save", applied) if applied else ("save_as_new", None)]
    if reopened:
        actions.append(("cancel", None))
    return actions


def _buttons(df, saved_names):
    roles = st.session_state._review_roles
    groups = st.session_state._review_groups
    names = st.session_state._review_group_names
    source = _applied_profile()
    blocked = review_blocking_reason(df, roles)
    actions = exit_actions(source, bool(st.session_state.get("_review_snapshot")))

    # Save-as uses adjacent slots for its button and name field.
    widths = [width for kind, _ in actions
              for width in ((1, 2) if kind == "save_as_new" else (1,))]
    cols = st.columns([*widths, 2])
    slots = iter(cols)
    for kind, profile in actions:
        if kind == "use":
            with next(slots):
                if st.button("Use for this upload", key="review_use", type="primary",
                             width="stretch", disabled=bool(blocked)):
                    _close_review()
        elif kind == "save":
            with next(slots):
                if st.button(f"💾 Save to {profile} & use",
                             width="stretch", disabled=bool(blocked)):
                    _save_and_close(profile, roles, groups, names)
        elif kind == "save_as_new":
            _save_as_new(next(slots), next(slots), roles, groups, names, blocked,
                         saved_names)
        else:
            with next(slots):
                if st.button("Cancel", key="review_cancel", width="stretch",
                             help="Discard these edits and go back"):
                    snapshot = st.session_state._review_snapshot
                    for key in _COPY_KEYS:
                        if key in snapshot:
                            st.session_state[key] = deepcopy(snapshot[key])
                        else:
                            st.session_state.pop(key, None)
                    st.session_state._review_previous_roles = dict(st.session_state._review_roles)
                    st.session_state._review_picks_stale = True
                    _bump_editor()
                    _close_review()
    if blocked:
        cols[-1].error(blocked)


def _save_as_new(button_slot, name_slot, roles, groups, names, blocked, saved_names):
    """Render a profile name and save button, confirming existing-name overwrites.

    Read the name field before the button handler, while placing it on the right.
    The field is outside a form, so Return retains the name without saving.
    Confirmation applies only while the trimmed input matches the armed name
    and that profile still exists. Keep the field mounted to retain its value.
    Existing names can be replaced even at the profile cap.

    Keep count/overwrite help on the button because the name field's hidden
    label also hides its tooltip.
    """
    with name_slot:
        new_name = st.text_input(
            "Profile name", key=f"review_save_as_name_{_file_gen()}",
            label_visibility="collapsed", placeholder="name this profile")
        typed = new_name.strip()
        armed = bool(typed) and typed in saved_names and \
            st.session_state.get(_OVERWRITE_ARMED) == typed
        if armed:
            # Escape the typed profile name while preserving the caption markup.
            st.caption(f"{code_span(typed)} already exists — press **Replace profile** to "
                       "overwrite its roles and groups, or change the name.")
    with button_slot:
        # Give initial-save and overwrite-confirm actions distinct widget identities.
        if st.button("⚠️ Replace profile" if armed else "💾 Save profile as",
                     key=f"review_save_as_{'confirm' if armed else 'new'}_{_file_gen()}",
                     width="stretch", disabled=bool(blocked),
                     help=f"{len(saved_names)} of {MAX_PROFILES} profiles saved. "
                          "An existing name overwrites that profile."):
            if not armed and typed in saved_names:
                st.session_state[_OVERWRITE_ARMED] = typed
                st.rerun()
            else:
                _save_and_close(new_name, roles, groups, names)


def _save_and_close(name, roles, groups, group_names):
    """Save the working copy and resume analysis only when the write succeeds."""
    try:
        error = save_working_copy(name, roles, groups, group_names=group_names)
    except OSError as exc:
        error = str(exc)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    st.session_state.pop(_OVERWRITE_ARMED, None)
    st.session_state._review_saved_as = name.strip()
    st.session_state._review_source = name.strip()
    st.session_state._review_known_cols = set(roles)
    st.session_state._review_baseline = _assignments()
    st.session_state._review_toast = f"Saved to {code_span(name.strip())} {happy_emoji}"
    _close_review()


def show_review_toast():
    """Show the toast a review action left, once.

    Review actions end with st.rerun(), and closing review can rerun the page again,
    so a toast shown in either interrupted run unmounts unseen. The open gate shows
    it before it stops the page; after closing, the page shows it once rendered.
    """
    toast = st.session_state.pop("_review_toast", "")
    if toast:
        st.toast(toast)


def _close_review():
    st.session_state._review_confirmed = True
    # Reopening starts with no row ticked.
    st.session_state._review_picks_stale = True
    st.session_state.pop("_review_snapshot", None)
    _decision()
    st.rerun()


def reopen_gate():
    """Reopen the current working copy and preserve analysis settings.

    Mark review opened to prevent exact-match auto-application, enable Cancel,
    and capture control keys before any review widgets render. The gate keeps
    those keys alive while analysis is hidden; the page clears the captured set
    after a resumed analysis render completes.
    """
    if st.session_state.get("_review_confirmed"):
        st.session_state._review_snapshot = deepcopy({
            key: st.session_state[key] for key in _COPY_KEYS if key in st.session_state})
    st.session_state._review_confirmed = False
    st.session_state._review_opened = True
    st.session_state._review_preserve_controls = analysis_control_keys(st.session_state)
