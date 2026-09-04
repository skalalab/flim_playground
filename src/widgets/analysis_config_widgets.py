import sys
from pathlib import Path

import streamlit as st

from src.column_roles import (
    ROLE_CATEGORICAL,
    ROLE_IGNORE,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    ROLES,
    code_span,
)
from src.config import (
    get_categorical_cols,
    get_fov_name_col,
    get_persistent_dir,
    get_unique_cell_id_col,
    load_config,
    save_config,
)

# Maximum number of profiles allowed. Every saved profile is read on every rerun: the
# whole config is parsed uncached to match an upload against all of them, and the chooser
# draws a row per profile in a page that does not scroll -- so the cap bounds a per-rerun
# cost and a screenful, not just a file.
#
# Measured on 60-column profiles: one parse costs ~0.14 ms per saved profile (0.16 ms at
# 1 profile, 2.8 ms at 20, linear), and `all_profile_columns` costs the same as
# `list_profiles` -- assembling the role maps is free beside the TOML parse. The gate
# threads one read down its whole call chain (`review_gate` -> `_render_gate` ->
# `_chooser` / `_buttons` / `_manage_profiles`), so it is **one** parse per rerun; the
# page's own accessors add two more. Deliberately not memoised: the gate writes this
# file from inside a rerun, so a stale profile list would be a worse bug than a slow one.
MAX_PROFILES = 20
# The gate section that renames and deletes profiles. Named here rather than in
# `review_table_widget`, which draws it, because the at-the-cap message below has to
# point at it and the import runs the other way.
MANAGE_LABEL = "Manage saved profiles"
# The chooser's own row, which stands for "no profile". Named here for the same reason:
# every write below has to refuse it as a profile name, and the import runs the other way.
AUTO_DETECT = "Auto-detect — start a new profile"

# Columns no file contains: the 1D/2D GMM and K-Means plots add them to the frame at plot
# time. They are categoricals wherever a categorical list is assembled -- that is what
# keeps get_features from offering a cluster number as a measurement.
_PLATFORM_CATEGORICALS = ("GMM_group", "2D_GMM_group", "k_means_cluster")


def _dedup(names):
    """The names in order, first occurrence kept, blanks dropped."""
    return list(dict.fromkeys(name for name in names if name))

def _get_analysis_config_path() -> Path:
    """Get the analysis config file path, handling both development and bundled app scenarios."""
    # Check if running as a PyInstaller bundle
    if getattr(sys, '_MEIPASS', None):
        # Running as bundled app - save config in the per-user location
        # get_persistent_dir() picks (outside the swappable app payload).
        # analysis_config.toml is not bundled, and nothing seeds it: the file appears
        # when the user's first Save writes a profile into it.
        return get_persistent_dir() / "analysis_config.toml"
    else:
        # Running in development mode - use analysis_config.toml in project root
        return Path(__file__).resolve().parent.parent.parent / "analysis_config.toml"

# Absolute path to the analysis config file - handles both dev and bundled scenarios
_ANALYSIS_CONFIG_PATH = _get_analysis_config_path()

def _migrate_old_config_to_profiles(cfg: dict) -> dict:
    """Bring a config read off disk up to the current shape.

    Two migrations, both idempotent, both applied on every read: a flat config becomes
    `profiles.default`, and a profile saved before the two keys below existed gets them.
    **This function never writes** -- it edits the dict the caller just parsed, and reads
    stop there. Every accessor already tolerates a missing key through a
    `.get(key, default)`, so the top-up buys a shape guaranteed in one place instead of
    defended at each reader, at no cost to the file. A *write* path (`_save_profile_config`,
    `rename_profile`, `delete_profile`) runs this first and then persists, so the shape
    becomes durable on the next save rather than on a page load.

    Both keys seed **empty**. Every seeded name is a claim about a column the file need
    not have: a seeded identifier rejects the file outright and a seeded categorical
    list quietly describes someone else's data.
    """
    # Check if old format exists
    if "profiles" not in cfg and any(key in cfg for key in ["unique_row_id_col", "fov_name_col", "categorical_cols", "feature_groups", "all_numerical_features"]):
        # Migrate to profile-based format
        profiles = {
            "default": {
                "unique_row_id_col": cfg.get("unique_row_id_col", ""),
                "fov_name_col": cfg.get("fov_name_col", ""),
                "categorical_cols": cfg.get("categorical_cols", []),
                "feature_groups": cfg.get("feature_groups", {}),
                "all_numerical_features": cfg.get("all_numerical_features", [])
            }
        }
        # Remove old keys
        for key in ["unique_row_id_col", "fov_name_col", "categorical_cols", "feature_groups", "all_numerical_features"]:
            cfg.pop(key, None)
        cfg["profiles"] = profiles
        cfg["current_profile"] = "default"

    for profile_cfg in cfg.get("profiles", {}).values():
        if isinstance(profile_cfg, dict):
            # A fresh [] per profile, never one shared default: a caller that mutates
            # the list it read would otherwise push that mutation into every profile.
            profile_cfg.setdefault("unique_row_id_col", "")
            profile_cfg.setdefault("categorical_cols", [])

    return cfg

def _get_current_profile() -> str:
    """The profile a Save, a rename or a delete last left in force, or "" for none.

    **Reading never creates one**, and must not: saving is the only way a profile is
    made, `_applied_profile` deliberately does not read this, and `working_copy_arguments`
    supplies the roles the analysis runs on. A profile minted by a read would not be
    inert either -- it spends one of MAX_PROFILES, and it knows no columns, so
    `ProfileFit.is_exact` needs its "shared must be non-empty" clause to stop such a
    profile claiming every file (two empty sets compare equal). That clause stays: a
    config written by an older build still holds the empty profile this no longer mints.

    So **no active profile is a reachable state**, and every caller must tolerate "".
    `_get_profile_config("")` answers `{}`, which every accessor reads through a
    `.get(key, default)`; and `_save_profile_config` is never reached with it, since the
    only writes are a validated Save.
    """
    if "current_profile" not in st.session_state:
        cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
        name = cfg.get("current_profile", "")
        st.session_state.current_profile = name if name in cfg.get("profiles", {}) else ""
    return st.session_state.current_profile

def _get_profile_config(profile_name: str | None = None) -> dict:
    """One profile's stored config, or `{}` when there is no such profile.

    Read-only, including for a name the config does not hold: nothing here inserts, so
    there is one answer to where profiles come from -- `save_working_copy`.
    """
    if profile_name is None:
        profile_name = _get_current_profile()

    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    return cfg.get("profiles", {}).get(profile_name, {})

def _save_profile_config(profile_name: str, profile_data: dict):
    """Save config for a specific profile."""
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    cfg = _migrate_old_config_to_profiles(cfg)

    if "profiles" not in cfg:
        cfg["profiles"] = {}

    cfg["profiles"][profile_name] = profile_data
    cfg["current_profile"] = profile_name
    save_config(cfg, _ANALYSIS_CONFIG_PATH)

def get_unique_row_id_col(use_data_extraction=True):
    if use_data_extraction:
        return get_unique_cell_id_col()
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("unique_row_id_col", "")

def get_fov_name_col_analysis(use_data_extraction=True):
    """The designated FOV column, which only the extraction branch has.

    A user's table may carry a field-of-view column, but it is an ordinary categorical
    there -- no role names it, so there is nothing to return and "" is the honest
    answer. Extraction is the branch that genuinely has one: config.toml names it and
    extraction always emits it, which is also why the missing-FOV warning fires there
    and nowhere else.
    """
    return get_fov_name_col() if use_data_extraction else ""

# No get_ignored_cols_analysis accessor: an ignored column reaches get_features from the
# review table's working copy (working_copy_arguments), never from the active profile --
# the file picks the profile, so the active one need not be the matched one. Extraction
# data has nothing to dismiss, and load_table leaves ignored_cols unset. The stored key
# is still read, by profile_column_roles below, as the ROLE_IGNORE half of the role map.


def get_categorical_cols_analysis(use_data_extraction=True):
    if use_data_extraction:
        data_extraction_categorical_cols = get_categorical_cols()
        fov_name_col = get_fov_name_col()
        # De-duplicated, and that is not tidiness. Both names are free text on the Home
        # page -- fov_name_col a text_input, the categoricals an accept_new_options
        # multiselect -- so typing "image_name" into both is an easy and reasonable
        # thing to do. get_features keeps every matching categorical by name, so a
        # repeat put the column into columns_to_keep twice and df[columns_to_keep]
        # returned a frame with two identical columns; every later df[fov_col] then
        # handed back a DataFrame instead of a Series.
        return _dedup(data_extraction_categorical_cols + [fov_name_col]
                      + list(_PLATFORM_CATEGORICALS))
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)

    # De-duplicated for the same reason as the branch above, against a different source:
    # here the repeat is already *in* the stored list, from a profile written by the old
    # free-text categorical multiselect or a hand-edited analysis_config.toml. Blanks go
    # too, since both identifier fields are optional and "" names no column.
    #
    # Assembled into a new list rather than appended to the stored one, which used to
    # mutate the list inside the parsed profile: harmless only because every call
    # re-parsed the file, so anything that memoises that read would start accumulating
    # the platform columns into the profile and write them back on the next Save.
    #
    # Legacy only: a profile saved while the FOV role existed stored that column in
    # fov_name_col as well as categorical_cols, and one migrated from the old flat
    # config may have it in fov_name_col alone. Folding it in here is what keeps such a
    # profile knowing the same columns it always did -- profile_column_roles does the
    # same, and the first Save rewrites the profile without the key.
    #
    # _PLATFORM_CATEGORICALS is what the 1D/2D GMM and K-Means plots add to the frame.
    return _dedup(list(profile_cfg.get("categorical_cols") or [])
                  + [profile_cfg.get("fov_name_col") or ""]
                  + list(_PLATFORM_CATEGORICALS))

def get_all_feature_groups():
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("feature_groups", {})


def profile_column_roles(profile_cfg=None):
    """The profile's columns as `{name: role}`, assembled from the stored lists.

    A view, not the storage format: analysis_config.toml keeps the parallel lists it
    has always had, so existing profiles load untouched and nothing migrates. This is
    the shape the review table edits and the shape matching compares.

    Blank identifiers name no column -- both are optional, and "" is a reachable value.
    """
    if profile_cfg is None:
        profile_cfg = _get_profile_config(_get_current_profile())

    by_role = {
        ROLE_ROW_ID: [profile_cfg.get("unique_row_id_col") or ""],
        # A stored fov_name_col is read as an ordinary categorical. There is no FOV
        # role to give it back to, and a profile written before the role was dropped
        # already lists that column in categorical_cols too -- so this only matters for
        # one migrated from the old flat config, where it could be in neither.
        ROLE_CATEGORICAL: ((profile_cfg.get("categorical_cols") or [])
                           + [profile_cfg.get("fov_name_col") or ""]),
        ROLE_NUMERICAL: profile_cfg.get("all_numerical_features") or [],
        ROLE_IGNORE: profile_cfg.get("ignored_cols") or [],
    }
    roles = {}
    # Lowest precedence first, so a higher role overwrites rather than being skipped.
    for role in reversed(ROLES):
        for col in by_role[role]:
            if col:
                roles[col] = role
    return roles


def apply_column_roles(profile_cfg, roles):
    """Write a `{name: role}` map back out to the profile's stored lists, in place.

    The inverse of profile_column_roles. No fov_name_col is written: a field-of-view
    column is an ordinary categorical here and lands in categorical_cols like the rest,
    which is what stringifies it, fills "N/A" and makes it filterable. Because the whole
    profile is replaced on Save, a legacy key simply stops existing after the first one.

    An unassigned identifier is blanked rather than left at its old value, or clearing
    Row ID in the table would silently keep the previous column.
    """
    row_ids = [col for col, role in roles.items() if role == ROLE_ROW_ID]
    cats = [col for col, role in roles.items() if role == ROLE_CATEGORICAL]

    profile_cfg["unique_row_id_col"] = row_ids[0] if row_ids else ""
    profile_cfg["categorical_cols"] = cats
    profile_cfg["all_numerical_features"] = [
        col for col, role in roles.items() if role == ROLE_NUMERICAL]
    profile_cfg["ignored_cols"] = [
        col for col, role in roles.items() if role == ROLE_IGNORE]
    return profile_cfg


def profile_roles_and_groups(name):
    """One named profile as `({column: role}, {column: group}, [group names])`.

    By name rather than "the active profile", because under this design the file picks
    the profile: the one an upload matched need not be the one that happens to be current.

    The names ride along because the mapping cannot express an empty group, and one read
    of the config serves all three -- see profile_group_names.
    """
    profile_cfg = _get_profile_config(name)
    return (profile_column_roles(profile_cfg), column_groups(profile_cfg),
            profile_group_names(profile_cfg))


def working_copy_arguments(roles, groups, group_names=None):
    """The review table's decision as the arguments interpret_table takes.

    Everything the analysis used to read from the active profile comes through here
    instead, which is what makes an unsaved edit take effect and what stops a file
    matching `pdl1` from taking its groups from whichever profile is current.

    Assembled through apply_column_roles rather than by re-deriving the lists, so the
    hand-off and a Save can never disagree about what a role means.
    """
    profile_cfg = {}
    apply_column_roles(profile_cfg, roles)
    apply_column_groups(profile_cfg, groups, group_names=group_names)
    categorical_cols = list(profile_cfg["categorical_cols"])
    for col in _PLATFORM_CATEGORICALS:
        if col not in categorical_cols:
            categorical_cols.append(col)
    return {
        "categorical_cols": categorical_cols,
        "unique_row_id_col": profile_cfg["unique_row_id_col"],
        "ignored_cols": profile_cfg["ignored_cols"],
        "feature_groups": profile_cfg["feature_groups"],
    }


def _reserved_name_error(name):
    """Why no profile may carry this name, or "".

    The chooser draws AUTO_DETECT as a row beside the profile names and maps a pick on
    it to "no profile". A profile holding that name would therefore draw a second row
    under the same widget key -- which raises, taking the whole gate down with it, so the
    file could not be opened at all -- and a pick on its own row would auto-detect
    instead of loading it. Checked on every write rather than in the box that types it:
    Save as and Rename are separate screens and would otherwise have to agree.
    """
    if name == AUTO_DETECT:
        return "That name belongs to the chooser's own Auto-detect row. Pick another."
    return ""


def list_profiles():
    """Every saved profile name, in the order the config file holds them."""
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    return list(cfg.get("profiles", {}))


def save_working_copy(name, roles, groups, group_names=None):
    """Write the review table's working copy to a profile. Returns "" or why it could not.

    The only write to disk in the whole upload flow, and the reason the working copy can
    be edited freely: a profile changes when this is called and at no other moment.

    `P` becomes exactly `F` -- the columns this file did not have are forgotten and the
    ones it added are recorded -- so the next upload of the same file is an exact match
    that skips the gate. Keeping a missing column would mean the profile just saved does
    not describe the file it was saved from.
    """
    name = (name or "").strip()
    if not name:
        return "A profile needs a name."
    reserved = _reserved_name_error(name)
    if reserved:
        return reserved
    existing = list_profiles()
    if name not in existing and len(existing) >= MAX_PROFILES:
        # Names only what is on screen, which rules out the chooser's rows: that list
        # holds candidates and is empty for a file sharing no column with any profile --
        # exactly the file most likely to be the one hitting the cap. The manage section
        # renders below the button this error appears beside, on every opening of the
        # gate, so "below" always holds. Its label is the constant above rather than a
        # repeated string: the sentence is only true while it names a section that exists
        # under that name.
        return (f"There are already {MAX_PROFILES} profiles. Type an existing name to "
                f"save over it, or delete one under **{MANAGE_LABEL}** below.")
    profile_cfg = {}
    apply_column_roles(profile_cfg, roles)
    apply_column_groups(profile_cfg, groups, group_names=group_names)
    _save_profile_config(name, profile_cfg)
    st.session_state.current_profile = name
    return ""


def rename_profile(old_name, new_name):
    """Rename a profile in place. Returns "" or why it could not.

    Insertion order is rebuilt rather than mutated so the panel's list does not reshuffle
    under the user -- a renamed profile stays where it was.
    """
    new_name = (new_name or "").strip()
    if not new_name:
        return "A profile needs a name."
    reserved = _reserved_name_error(new_name)
    if reserved:
        return reserved
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    profiles = cfg.get("profiles", {})
    if old_name not in profiles:
        return f"There is no profile called {code_span(old_name)}."
    if new_name == old_name:
        return ""
    if new_name in profiles:
        return f"A profile called {code_span(new_name)} already exists."
    cfg["profiles"] = {(new_name if name == old_name else name): data
                       for name, data in profiles.items()}
    if cfg.get("current_profile") == old_name:
        cfg["current_profile"] = new_name
        st.session_state.current_profile = new_name
    save_config(cfg, _ANALYSIS_CONFIG_PATH)
    return ""


def delete_profile(name):
    """Delete a profile. Returns "" or why it could not.

    Deleting the last one is allowed, unlike the old panel: a profile no longer has to
    exist before a file can be loaded, so "start over" is delete and nothing else. What
    remains is an empty profile, which ProfileFit.is_exact refuses to match any file with.
    """
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        return f"There is no profile called {code_span(name)}."
    del profiles[name]
    if cfg.get("current_profile") == name:
        # "" when that was the last one, not the name "default" -- inventing a profile
        # is exactly what deleting the last one is asking not to happen.
        survivor = next(iter(profiles), "")
        cfg["current_profile"] = survivor
        st.session_state.current_profile = survivor
    save_config(cfg, _ANALYSIS_CONFIG_PATH)
    return ""


def column_groups(profile_cfg=None):
    """The profile's groups as `{column: group}` -- the review table's Group column.

    The inverse view of the stored `{group: [columns]}`, mirroring what
    profile_column_roles does for roles. A column in no group is simply absent, as an
    ungrouped column has always been.
    """
    if profile_cfg is None:
        profile_cfg = _get_profile_config(_get_current_profile())
    return {col: group
            for group, cols in (profile_cfg.get("feature_groups") or {}).items()
            for col in cols}


def profile_group_names(profile_cfg=None):
    """The profile's group names, in the stored order, the empty ones included.

    The stored keys, never the values of `column_groups`' inverted view: a group with no
    members is invisible in the mapping, so reading the names back off it drops exactly
    the groups `apply_column_groups`' `group_names` argument exists to keep. An empty
    group is a name the user chose -- for a group whose columns this file does not have,
    or one made a moment before the columns to fill it -- and it survives a save only if
    it also survives the load.
    """
    if profile_cfg is None:
        profile_cfg = _get_profile_config(_get_current_profile())
    return list(profile_cfg.get("feature_groups") or {})


def apply_column_groups(profile_cfg, mapping, group_names=None):
    """Write a `{column: group}` map back to the stored `{group: [columns]}`, in place.

    `group_names` carries the groups the user has made but not filled, which the mapping
    alone cannot express -- a group with no members is invisible in it. Passing the
    review table's full list is what lets an empty group survive a Save, and it fixes the
    group order, which is the order the feature pickers show.
    """
    names = list(group_names) if group_names is not None else []
    for group in mapping.values():
        if group not in names:
            names.append(group)
    profile_cfg["feature_groups"] = {
        name: [col for col, group in mapping.items() if group == name] for name in names}
    return profile_cfg


def all_profile_columns():
    """`{profile name: known columns}` for every saved profile.

    What matching compares an uploaded file against. Reads every profile rather than
    the active one, because the file picks the profile here -- nothing is preselected,
    so a profile that does not fit simply is not the match and cannot be damaged by
    an upload that disagrees with it.
    """
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    return {name: profile_known_columns(profile_cfg)
            for name, profile_cfg in cfg.get("profiles", {}).items()}


def profile_known_columns(profile_cfg=None):
    """Every column this profile has seen -- `P` in the matching rule.

    Includes the ignored ones, which is the entire point: without them a profile
    saved from a file containing `notes` knows three columns while that same file
    has four, so re-uploading it would never match.
    """
    return set(profile_column_roles(profile_cfg))




