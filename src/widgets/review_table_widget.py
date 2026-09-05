"""Column review UI for editing a session-local working copy.

Profile matching and role rules live in ``profile_matching`` and ``column_roles``.
Review hides analysis until Save or Cancel resumes it. Edited roles and groups
reach the saved profile only through ``save_working_copy``.
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
from src.widgets.analysis_widget_state import analysis_control_keys, preserve_analysis_controls

GROUP_SECTION = "Feature group management"
GROUP_HELP = ("Each group becomes one dropdown in the feature pickers, so features are "
              "chosen from a few short lists instead of one long one. Measurements left "
              "ungrouped are collected under Uncategorized Features.")

# Clear the working copy and review state together when the file fingerprint changes.
_STATE_KEYS = (
    "_review_fingerprint", "_review_roles", "_review_groups", "_review_group_names",
    "_review_source", "_review_confirmed", "_review_previous_roles", "_review_known_cols",
    "_review_notices", "_review_opened", "_review_saved_as", "_review_numeric_cols",
    "_review_reopened", "_review_chooser", "_review_delete_armed", "_review_manage_open",
    "_review_configured_row_id", "_review_picks_stale", "_review_overwrite_armed",
    "_review_preserve_controls",
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
# Selection, new badge, name, role, group, preview. The group slot is wider
# than the role slot to fit UNGROUPED_LABEL.
_ROW_WEIGHTS = (1, 1, 5, 3, 4, 4)
# Include widget gaps and padding; cap tall tables so Save stays near the rows.
_ROW_HEIGHT, _HEADER_HEIGHT, _MAX_TABLE_HEIGHT = 56, 78, 470

_GEN = "_review_editor_gen"
# File-scoped keys preserve selections and typed names through editor corrections
# while resetting them when the file fingerprint changes.
_FILE_GEN = "_review_file_gen"


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


def _picked_columns(df):
    """Return selected measurements in file order.

    Read before rendering rows to include current clicks. Filter by role because
    a hidden checkbox's session key can survive for one rerun.
    """
    roles = st.session_state._review_roles
    return [col for col in df.columns
            if roles.get(col) == ROLE_NUMERICAL
            and st.session_state.get(_pick_key(col), False)]


def _clear_picks(columns):
    """Clear selections before their widgets render.

    Assignment updates widget state; deleting a key can restore the browser's
    previous value.
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
    # Preserve stored group order and empty groups before adding newly detected groups.
    # The reserved ungrouped label cannot also name a real group.
    names = [name for name in dict.fromkeys(profile_names) if name != UNGROUPED_LABEL]
    for col in df.columns:
        group = groups.get(col)
        if group and group not in names:
            names.append(group)
    st.session_state._review_group_names = names
    st.session_state._review_source = picked
    # A previously saved name no longer identifies this replacement working copy.
    st.session_state.pop("_review_saved_as", None)
    st.session_state._review_previous_roles = dict(roles)
    st.session_state._review_known_cols = set(profile_roles)
    # Reuse the numeric-column set from role detection; coercing again is expensive.
    st.session_state._review_numeric_cols = numeric_cols
    # Cancel calls this below rendered rows. Defer checkbox writes until the next
    # _group_section call, before those widgets render.
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
    name = _applied_profile() or "Auto-detected"
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

    An exact match auto-applies only on initial entry and when its roles are valid.
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
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        st.session_state._review_fingerprint = fingerprint
        # Reset file-scoped widget identities so selections and names cannot carry over.
        st.session_state[_FILE_GEN] = _file_gen() + 1
        _bump_editor()

    if not st.session_state.get("_review_confirmed") and not st.session_state.get("_review_opened"):
        profiles = all_profile_columns()
        matched = exact_match(set(df.columns), profiles)
        if matched:
            _load_working_copy(df, matched)
            # A matching column set can still contain an invalid Row ID. Show review
            # when the loaded roles would block analysis.
            st.session_state._review_confirmed = not review_blocking_reason(
                df, st.session_state._review_roles)

    control_keys = st.session_state.get("_review_preserve_controls")
    if control_keys:
        preserve_analysis_controls(st.session_state, control_keys)

    if st.session_state.get("_review_confirmed"):
        return _decision()

    st.session_state._review_opened = True
    # Keep the chooser's visibility fixed for the whole opening, even if a pick
    # changes whether a profile matches exactly.
    if "_review_chooser" not in st.session_state:
        if profiles is None:
            profiles = all_profile_columns()
        st.session_state._review_chooser = chooser_is_needed(
            _applied_profile(), set(df.columns), profiles)
    _render_gate(uploaded_file, df, profiles)
    # Review hides analysis, including on reopen. Save and Cancel rerun before
    # a confirmed decision reaches the page.
    return None


def _render_gate(uploaded_file, df, profiles=None):
    """Render profile choice, column editing, save controls, and profile management.

    Reuse profiles supplied by the gate. When the chooser is hidden, only
    profile names are needed.
    """
    name = getattr(uploaded_file, "name", "the uploaded file")
    # Render the filename literally inside the Markdown caption.
    st.caption(f"Read {len(df.columns)} columns × {len(df):,} rows from {code_span(name)}")

    if st.session_state.get("_review_chooser", True):
        if profiles is None:
            profiles = all_profile_columns()
        saved_names = list(profiles)
        _chooser(df, profiles)
        # A chooser click loads _review_source before rerunning.
        if st.session_state.get("_review_source") is None:
            return
    else:
        saved_names = list(profiles) if profiles is not None else list_profiles()

    for notice in st.session_state.pop("_review_notices", []):
        st.info(notice)
    _group_section(df)
    st.markdown(
        "**Column preview**",
        help="Use each column's **Role** dropdown to choose how it is used:\n\n"
             "- **Row ID**: one column of unique identifiers with no missing values. "
             "Optional; row numbers are generated if none is assigned.\n"
             "- **Categorical**: labels for grouping or filtering, such as treatment, "
             "donor, or batch, even when encoded as numbers.\n"
             "- **Numerical**: measurements to plot or analyze, such as lifetime "
             "or cell area.\n"
             "- **Ignore**: exclude the column from analysis.\n\n"
             "Only Numerical columns can be assigned to feature groups.",
    )
    _editor(df)
    # Use the corrected roles to announce row numbers before they are generated.
    numbering = row_id_notice(st.session_state._review_roles)
    if numbering:
        st.info(numbering)
    _buttons(df, saved_names)
    _manage_profiles(saved_names)


def _chooser(df, profiles):
    """Show candidates in ranked order and load only an explicit pick.

    The source lives in session state independently of the buttons, so other
    actions can replace the working copy after the chooser renders.
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
        # The table and save controls wait for a chooser option to be selected.
        st.caption("Ranked by how many of this file's columns each profile already "
                   "knows; profiles sharing none are left out. Pick one to fill in "
                   "the table below.")
    elif len(options) > 1:
        st.caption("Changing this rebuilds the table below and discards your edits.")


def _pick_row(df, name, label, picked):
    """Load a clicked chooser option and rerun to update every row's selection."""
    if st.button(label, key=f"review_pick_{name}",
                 type="primary" if name == picked else "tertiary") and name != picked:
        _load_working_copy(df, name)
        st.rerun()


def _manage_profiles(saved_names):
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
                    _delete_confirm(name)
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


def _delete_confirm(name):
    """Confirm deletion in place for the profile named by ``_DELETE_ARMED``.

    The armed profile name is server state, so confirmation cannot move to a
    neighboring row after deleting a profile.
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
    """Rename a profile and update both working-copy references to its name."""
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
    """Delete a profile and reset review if it supplied the current working copy.

    Check both applied and selected names because Save-as can leave them
    different. Clearing the copy also restores the chooser and removes Cancel:
    there is no saved decision to resume once its profile is deleted.
    """
    error = delete_profile(name)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    # Clear the deleted profile from confirmation state before rerunning.
    st.session_state.pop(_DELETE_ARMED, None)
    # Keep management open when the changed row list remounts it.
    st.session_state[_MANAGE_OPEN] = True
    if name in (_applied_profile(), st.session_state.get("_review_source")):
        for key in ("_review_source", "_review_saved_as", "_review_roles", "_review_groups",
                    "_review_group_names", "_review_known_cols", "_review_previous_roles",
                    "_review_reopened", "_review_preserve_controls"):
            st.session_state.pop(key, None)
        st.session_state._review_chooser = True
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
            # Render selection in the first cell after reading the current role,
            # so only measurements can be selected for bulk grouping.
            if measurement:
                cells[0].checkbox(f"Select {col}", key=_pick_key(col),
                                  label_visibility="collapsed",
                                  help="Tick to include in a bulk group assignment")
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


def _group_section(df):
    """Render group creation and assignment controls above the column rows.

    The shared name field serves Add and Rename. Delete, Rename, and Apply use
    the selected group. This section must run before the row checkboxes because
    it writes their session-state values.
    """
    # Apply deferred resets before the row checkboxes render.
    if st.session_state.pop("_review_picks_stale", False):
        _clear_picks(df.columns)
    roles = st.session_state._review_roles
    numeric = [col for col in df.columns if roles.get(col) == ROLE_NUMERICAL]
    if not numeric:
        # Only measurements can be grouped.
        return
    picked = _picked_columns(df)
    names = st.session_state._review_group_names
    gen = st.session_state[_GEN]

    # Prune a deleted or renamed destination before rendering its keyed selectbox.
    options = [NO_GROUP] + names
    target_key = f"review_group_target_{gen}"
    if st.session_state.get(target_key, NO_GROUP) not in options:
        st.session_state[target_key] = NO_GROUP

    st.markdown(f"**{GROUP_SECTION}**", help=GROUP_HELP)
    # Match column counts and total weights so the name field aligns with the
    # destination despite the narrow Delete/Rename slots.
    create = st.columns([3, 2, 1, 1, 2], vertical_alignment="center")
    act = st.columns([3, 1, 1, 2, 2], vertical_alignment="center")

    new_name = create[0].text_input(
        "Group name", key=f"review_group_name_{gen}", label_visibility="collapsed",
        placeholder="group name")
    if create[1].button("➕ Add", key="review_add_group", width="stretch",
                        help="Make an empty group with the name on the left"):
        _add_group(new_name)

    # Omit index: a seeded session value plus an explicit default triggers a warning.
    # An unseeded selectbox defaults to NO_GROUP at slot 0.
    target = act[0].selectbox(
        "Group", options, key=target_key, label_visibility="collapsed",
        format_func=_group_label,
        help=f"The group Delete, Rename and Apply act on. {UNGROUPED_LABEL} is not a "
             "group: applying it takes the ticked rows out of theirs.")
    # The ungrouped slot cannot be renamed or deleted.
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
    # Keep Select all outside the table header so it stays visible while rows scroll.
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
    """Assign selected columns to an existing group, or remove their assignments.

    The next render enforces role invariants, including removal of groups from
    columns that have become non-numerical.
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
    # Clear completed selections so offscreen rows are not included in the next action.
    _clear_picks(columns)
    _bump_editor()
    # Preserve the destination for another assignment. The new generation key
    # can be seeded because it has not rendered yet.
    st.session_state[f"review_group_target_{st.session_state[_GEN]}"] = group
    st.rerun()


def _group_name_error(name, allow_existing=False):
    """Validate Add/Rename names, allowing existing names only for merges."""
    # Reserve both ungrouped spellings to avoid indistinguishable dropdown options.
    if name in (NO_GROUP, UNGROUPED_LABEL):
        return f"{code_span(name)} is how the table shows a column with no group."
    if not allow_existing and name in st.session_state._review_group_names:
        return f"There is already a group called {code_span(name)}."
    return ""


def _add_group(name):
    """Create an empty group and select it as the bulk-assignment destination.

    Re-keying gives the destination an unrendered key that can be seeded safely.
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
    """Rename a group, merging its columns if the destination already exists."""
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
    # Keep the renamed group selected under the new widget key.
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
    """Offer one save action, plus Cancel when reopening a saved decision.

    An applied profile is updated; an auto-detected copy needs a profile name.
    Initial review cannot finish without saving.
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

    # Save-as uses adjacent slots for its button and name field.
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
                    # Restore saved roles and groups before resuming analysis.
                    _load_working_copy(df, source)
                    st.session_state._review_confirmed = True
                    st.rerun()
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
                     type="primary", width="stretch", disabled=bool(blocked),
                     help=f"{len(saved_names)} of {MAX_PROFILES} profiles saved. "
                          "An existing name overwrites that profile."):
            if not armed and typed in saved_names:
                st.session_state[_OVERWRITE_ARMED] = typed
                st.rerun()
            else:
                _save_and_close(new_name, roles, groups, names)


def _save_and_close(name, roles, groups, group_names):
    """Save the working copy and resume analysis only when the write succeeds."""
    error = save_working_copy(name, roles, groups, group_names=group_names)
    if error:
        st.error(f"{error} {sad_emoji}")
        return
    st.session_state.pop(_OVERWRITE_ARMED, None)
    st.session_state._review_saved_as = name.strip()
    st.session_state._review_confirmed = True
    st.toast(f"Saved to {code_span(name.strip())} {happy_emoji}")
    st.rerun()


def reopen_gate():
    """Reopen the current working copy and preserve analysis settings.

    Mark review opened to prevent exact-match auto-application, enable Cancel,
    and capture control keys before any review widgets render. The gate keeps
    those keys alive while analysis is hidden; the page clears the captured set
    after a resumed analysis render completes.
    """
    st.session_state._review_confirmed = False
    # Auto-apply can return before setting this flag; mark it here to keep review open.
    st.session_state._review_opened = True
    st.session_state._review_reopened = True
    st.session_state._review_chooser = chooser_is_needed(
        _applied_profile(), set(st.session_state.get("_review_roles") or {}),
        all_profile_columns())
    st.session_state._review_preserve_controls = analysis_control_keys(st.session_state)
