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
    unstorable_name_error,
)

# Bound the profile chooser's size and the cost of reading profiles on each rerun.
MAX_PROFILES = 20
# Shared with review_table_widget so validation messages name the visible controls
# without introducing a circular import.
MANAGE_LABEL = "Manage saved profiles"
# Reserved chooser label representing no saved profile.
AUTO_DETECT = "Auto-detect — start a new profile"

# Clustering plots add these categorical columns. Uploaded columns with the same
# names retain their reviewed roles in working_copy_arguments.
_PLATFORM_CATEGORICALS = ("GMM_group", "2D_GMM_group", "k_means_cluster")


def _dedup(names):
    """The names in order, first occurrence kept, blanks dropped."""
    return list(dict.fromkeys(name for name in names if name))

def _get_analysis_config_path() -> Path:
    """Use persistent storage for bundles and the project root for development."""
    if getattr(sys, '_MEIPASS', None):
        # The first profile save creates this file outside the app payload.
        return get_persistent_dir() / "analysis_config.toml"
    else:
        return Path(__file__).resolve().parent.parent.parent / "analysis_config.toml"

# Resolved once for the current runtime.
_ANALYSIS_CONFIG_PATH = _get_analysis_config_path()

def _migrate_old_config_to_profiles(cfg: dict) -> dict:
    """Normalize profile structure in place without writing to disk.

    Wrap flat analysis settings in profiles.default and seed missing identifier
    and categorical keys with empty values. This is idempotent; only an explicit
    save persists the result. Empty defaults make no assumptions about file columns.
    """
    if "profiles" not in cfg and any(key in cfg for key in ["unique_row_id_col", "fov_name_col", "categorical_cols", "feature_groups", "all_numerical_features"]):
        profiles = {
            "default": {
                "unique_row_id_col": cfg.get("unique_row_id_col", ""),
                "fov_name_col": cfg.get("fov_name_col", ""),
                "categorical_cols": cfg.get("categorical_cols", []),
                "feature_groups": cfg.get("feature_groups", {}),
                "all_numerical_features": cfg.get("all_numerical_features", [])
            }
        }
        for key in ["unique_row_id_col", "fov_name_col", "categorical_cols", "feature_groups", "all_numerical_features"]:
            cfg.pop(key, None)
        cfg["profiles"] = profiles
        cfg["current_profile"] = "default"

    for profile_cfg in cfg.get("profiles", {}).values():
        if isinstance(profile_cfg, dict):
            # Each profile needs its own mutable categorical list.
            profile_cfg.setdefault("unique_row_id_col", "")
            profile_cfg.setdefault("categorical_cols", [])

    return cfg

def _get_current_profile() -> str:
    """Return the session's current profile name, or "" when none is selected.

    Initialize from disk once per session without creating a profile. Upload
    matching and analysis use the working copy, independently of this selection.
    """
    if "current_profile" not in st.session_state:
        cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
        name = cfg.get("current_profile", "")
        st.session_state.current_profile = name if name in cfg.get("profiles", {}) else ""
    return st.session_state.current_profile

def _get_profile_config(profile_name: str | None = None) -> dict:
    """Read a profile's stored config, returning {} without creating it if absent."""
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
    """Return the extraction FOV column, or "" for a user table.

    User-table FOV columns use the ordinary Categorical role.
    """
    return get_fov_name_col() if use_data_extraction else ""

def get_categorical_cols_analysis(use_data_extraction=True):
    if use_data_extraction:
        data_extraction_categorical_cols = get_categorical_cols()
        fov_name_col = get_fov_name_col()
        # Free-text settings may repeat the FOV name. Keep each column once so
        # downstream df[name] access returns a Series.
        return _dedup(data_extraction_categorical_cols + [fov_name_col]
                      + list(_PLATFORM_CATEGORICALS))
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)

    # Include a stored fov_name_col as categorical for compatibility. Build a new
    # list, removing repeated and blank names without mutating profile settings.
    return _dedup(list(profile_cfg.get("categorical_cols") or [])
                  + [profile_cfg.get("fov_name_col") or ""]
                  + list(_PLATFORM_CATEGORICALS))

def get_all_feature_groups():
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("feature_groups", {})


def profile_column_roles(profile_cfg=None):
    """Return `{name: role}` from stored lists, using ROLES precedence for overlaps.

    Blank names are omitted. The returned view is used by review and matching;
    the stored profile is unchanged.
    """
    if profile_cfg is None:
        profile_cfg = _get_profile_config(_get_current_profile())

    by_role = {
        ROLE_ROW_ID: [profile_cfg.get("unique_row_id_col") or ""],
        # Support stored FOV names that are absent from categorical_cols.
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
    """Write `{name: role}` to the profile's role lists in place.

    FOV columns belong in categorical_cols. Set unique_row_id_col to "" when no
    Row ID is assigned so clearing that role also clears the stored identifier.
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
    """Read a named profile as `({column: role}, {column: group}, [group names])`.

    The named profile may differ from the current one. Group names are returned
    separately to preserve empty groups, using the same config read.
    """
    profile_cfg = _get_profile_config(name)
    return (profile_column_roles(profile_cfg), column_groups(profile_cfg),
            profile_group_names(profile_cfg))


def working_copy_arguments(roles, groups, group_names=None):
    """Convert the review working copy into interpret_table arguments.

    Convert supplied roles and groups independently of the current profile.
    Reuse the save serializers so analysis and persistence interpret them alike.
    """
    profile_cfg = {}
    apply_column_roles(profile_cfg, roles)
    apply_column_groups(profile_cfg, groups, group_names=group_names)
    categorical_cols = list(profile_cfg["categorical_cols"])
    # Reserve plot-generated categoricals only for names absent from the file.
    # A file column with the same name must retain its reviewed role.
    for col in _PLATFORM_CATEGORICALS:
        if col not in categorical_cols and col not in roles:
            categorical_cols.append(col)
    return {
        "categorical_cols": categorical_cols,
        "unique_row_id_col": profile_cfg["unique_row_id_col"],
        "ignored_cols": profile_cfg["ignored_cols"],
        "feature_groups": profile_cfg["feature_groups"],
    }


def _reserved_name_error(name):
    """Reject the chooser's reserved label or a name TOML cannot preserve.

    Save and Rename share this check to keep profile lookup and chooser keys valid.
    """
    if name == AUTO_DETECT:
        return "That name belongs to the chooser's own Auto-detect row. Pick another."
    return unstorable_name_error(name)


def list_profiles():
    """Every saved profile name, in the order the config file holds them."""
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    return list(cfg.get("profiles", {}))


def save_working_copy(name, roles, groups, group_names=None):
    """Save the working copy as a profile, returning "" or a validation error.

    Replace the profile with this file's roles and groups, dropping absent columns
    and recording new ones. Editing the working copy alone does not write to disk.
    """
    name = (name or "").strip()
    if not name:
        return "A profile needs a name."
    reserved = _reserved_name_error(name)
    if reserved:
        return reserved
    existing = list_profiles()
    if name not in existing and len(existing) >= MAX_PROFILES:
        # The management section lists all profiles, even when none match this file.
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
    """Delete a profile, returning "" or an error if it does not exist.

    Deleting the last profile leaves no saved profiles and clears current_profile.
    """
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        return f"There is no profile called {code_span(name)}."
    del profiles[name]
    if cfg.get("current_profile") == name:
        # No replacement profile is created when the last one is deleted.
        survivor = next(iter(profiles), "")
        cfg["current_profile"] = survivor
        st.session_state.current_profile = survivor
    save_config(cfg, _ANALYSIS_CONFIG_PATH)
    return ""


def column_groups(profile_cfg=None):
    """Invert stored `{group: [columns]}` into `{column: group}`, omitting ungrouped columns."""
    if profile_cfg is None:
        profile_cfg = _get_profile_config(_get_current_profile())
    return {col: group
            for group, cols in (profile_cfg.get("feature_groups") or {}).items()
            for col in cols}


def profile_group_names(profile_cfg=None):
    """Return group names in stored order, including empty groups.

    Read the stored keys: empty groups are absent from column_groups' inverse map.
    """
    if profile_cfg is None:
        profile_cfg = _get_profile_config(_get_current_profile())
    return list(profile_cfg.get("feature_groups") or {})


def apply_column_groups(profile_cfg, mapping, group_names=None):
    """Write a `{column: group}` map back to the stored `{group: [columns]}`, in place.

    `group_names` preserves empty groups and their order. Additional names in the
    mapping follow in first-occurrence order.
    """
    names = list(group_names) if group_names is not None else []
    for group in mapping.values():
        if group not in names:
            names.append(group)
    profile_cfg["feature_groups"] = {
        name: [col for col, group in mapping.items() if group == name] for name in names}
    return profile_cfg


def all_profile_columns():
    """Return `{profile name: known columns}` for matching against all saved profiles."""
    cfg = _migrate_old_config_to_profiles(load_config(_ANALYSIS_CONFIG_PATH))
    return {name: profile_known_columns(profile_cfg)
            for name, profile_cfg in cfg.get("profiles", {}).items()}


def profile_known_columns(profile_cfg=None):
    """Return all known columns, including ignored ones so the same file can match."""
    return set(profile_column_roles(profile_cfg))


