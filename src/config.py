import toml
import sys
from pathlib import Path
from typing import Optional

def get_persistent_dir() -> Path:
    """Directory for runtime-writable config, beside the app the user launched.

    In a macOS .app bundle sys.executable is
    ``<dir>/Flim-Playground.app/Contents/MacOS/Flim-Playground``; we walk up out
    of the bundle to ``<dir>`` so config lands next to the .app — mirroring where
    Windows puts it next to the .exe. Writing inside ``Contents/`` would hide the
    file from Finder and break the code-signature seal. On Windows/Linux the exe
    isn't in that layout, so we fall back to the executable's own directory.
    """
    exe = Path(sys.executable)
    if exe.parent.name == "MacOS" and exe.parent.parent.name == "Contents":
        return exe.parent.parent.parent.parent  # directory containing the .app
    return exe.parent


def _get_config_path() -> Path:
    """Get the config file path, handling both development and bundled app scenarios."""
    # Check if running as a PyInstaller bundle
    if getattr(sys, '_MEIPASS', None):
        # Running as bundled app - persist config next to the app.
        # config.toml is no longer bundled; main.py seeds defaults on first run.
        return get_persistent_dir() / "config.toml"
    else:
        # Running in development mode - use config.toml in project root
        return Path(__file__).resolve().parent.parent / "config.toml"

# Absolute path to the project-level config file
_CONFIG_PATH = _get_config_path()

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load and return the TOML configuration as a dict.

    Returns an empty dict if the file is missing or unparsable so that the
    calling code can supply sensible fall-backs.
    """
    path_to_load = _CONFIG_PATH if config_path is None else config_path
    try:
        return toml.load(path_to_load)
    except (FileNotFoundError, toml.TomlDecodeError):
        return {}

def save_config(cfg: dict, config_path: Optional[Path] = None) -> None:
    """Persist *cfg* to disk, overwriting the previous config file."""
    path_to_save = _CONFIG_PATH if config_path is None else config_path
    # Ensure the parent directory exists (helpful when running tests, etc.)
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    with path_to_save.open("w", encoding="utf-8") as fh:
        toml.dump(cfg, fh)

# ---------------------------------------------------------------------------
# Multi-profile support
#
# The extraction config now stores named profiles, mirroring analysis_config:
#   current_profile = "default"
#   [profiles.<name>]   # the entire legacy flat config lives in here
#
# ``current_profile`` is the single source of truth and is read from disk so
# this module stays Streamlit-free; the Configuration page (main.py) persists
# it immediately on every switch/create/delete (before st.rerun()). Do NOT read
# the active profile from st.session_state here.
# ---------------------------------------------------------------------------

def _migrate_extraction_config_to_profiles(cfg: dict) -> dict:
    """Wrap a legacy flat extraction config under ``profiles.default``.

    Idempotent: a config that already has a ``profiles`` key is returned
    unchanged. An empty dict (missing/unparsable file) is left untouched so the
    caller can seed defaults. The new structure is built in memory and returned;
    the input is not mutated.
    """
    if "profiles" in cfg:
        return cfg
    if not cfg:
        return cfg
    # Move every existing top-level key into a single "default" profile.
    default_profile = {k: v for k, v in cfg.items() if k != "current_profile"}
    return {"current_profile": "default", "profiles": {"default": default_profile}}

def _load_active_profile_cfg(config_path: Optional[Path] = None) -> dict:
    """Return the active profile's config sub-dict.

    The sub-dict has the same flat shape as the pre-profile config, so the
    accessor functions below read it exactly as they read the legacy config.
    """
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    current = cfg.get("current_profile", "default")
    return cfg.get("profiles", {}).get(current, {})

def get_current_profile_name(config_path: Optional[Path] = None) -> str:
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    return cfg.get("current_profile", "default")

def list_profiles(config_path: Optional[Path] = None) -> list:
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    return list(cfg.get("profiles", {}).keys())

def set_current_profile(name: str, config_path: Optional[Path] = None) -> None:
    """Switch the active profile, creating it (empty) if it does not exist."""
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    cfg.setdefault("profiles", {}).setdefault(name, {})
    cfg["current_profile"] = name
    save_config(cfg, config_path)

def create_profile(name: str, config_path: Optional[Path] = None) -> None:
    """Create a new blank profile and make it current.

    The profile is stored empty; main.py seeds it with app defaults on render
    and persists those on the first "Update Configuration" click.
    """
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    cfg.setdefault("profiles", {})
    if name not in cfg["profiles"]:
        cfg["profiles"][name] = {}
    cfg["current_profile"] = name
    save_config(cfg, config_path)

def delete_profile(name: str, config_path: Optional[Path] = None) -> None:
    """Delete a profile; if it was current, switch to the first remaining one.

    There is always at least one profile: deleting the last one recreates an
    empty ``default``.
    """
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    profiles = cfg.setdefault("profiles", {})
    profiles.pop(name, None)
    if cfg.get("current_profile") == name or cfg.get("current_profile") not in profiles:
        remaining = list(profiles.keys())
        if remaining:
            cfg["current_profile"] = remaining[0]
        else:
            profiles["default"] = {}
            cfg["current_profile"] = "default"
    save_config(cfg, config_path)

def get_unique_cell_id_col() -> str:
    cfg = _load_active_profile_cfg()
    return cfg.get("unique_cell_id_col", "cell_id")

def get_fov_name_col() -> str:
    cfg = _load_active_profile_cfg()
    return cfg.get("fov_name_col", "image_name")

def get_input_types(channel_keys: list) -> dict:
    cfg = _load_active_profile_cfg()
    input_types = {}
    for channel_key in channel_keys:
        input_types[channel_key] = cfg.get(channel_key, {}).get("input_type", None)
    return input_types

def get_default_file_suffixes(channel_key: str, input_type: str, selected_feature_extractors: list) -> dict:
    cfg = _load_active_profile_cfg()
    filtered_file_suffixes = {}
    file_suffixes = cfg.get(channel_key, {}).get(input_type, {}).get("input_suffixes", {})
    fit_free_calibration = cfg.get(input_type, {}).get("fit_free_calibration", "")
    for file_type in file_suffixes.keys():
        # Only include Fluorescence Lifetime Standard when fit free uses Fluorescence Lifetime Standard and this channel does fit free
        if file_type == "Decay" and "prefitted" in input_type and len(selected_feature_extractors) == 1 and "Lifetime fit" in selected_feature_extractors:
            continue
        if file_type == "Fluorescence Lifetime Standard":
            if not ("Lifetime fit free" in selected_feature_extractors and fit_free_calibration == "Fluorescence Lifetime Standard"):
                continue
        # skip a bunch of things 
        if file_type == "SPCImage t1" and "Lifetime fit" not in selected_feature_extractors:
            continue
        # skip IRF if no Lifetime extractors OR if prefitted and no fit free extractors
        if file_type == "IRF" and (not any("Lifetime" in extractor for extractor in selected_feature_extractors) or 
                                        ("prefitted" in input_type and "Lifetime fit free" not in selected_feature_extractors)):
            continue
        if file_type == "IRF" and ("Lifetime fit" not in selected_feature_extractors or "prefitted" in input_type) and "Lifetime fit free" in selected_feature_extractors and fit_free_calibration == "Fluorescence Lifetime Standard":
            continue
        filtered_file_suffixes[file_type] = file_suffixes[file_type]
    return filtered_file_suffixes

def get_channel_names() -> dict:
    cfg = _load_active_profile_cfg()
    num_channels = cfg.get("num_channels", 0)
    channel_names = {}
    for i in range(num_channels):
        channel_key = f"ch{i+1}"
        channel_names[channel_key] = cfg.get(channel_key, {}).get("channel_name", channel_key)
    return channel_names

def get_spc_output_suffix() -> dict:
    cfg = _load_active_profile_cfg()
    spc_output_suffix = cfg.get("spc_output_suffix", {})
    return spc_output_suffix

def get_num_components(input_types: dict, channel_keys: list) -> dict:
    cfg = _load_active_profile_cfg()
    num_components = {}
    for channel_key in channel_keys:
        input_type = input_types[channel_key]
        num_components[channel_key] = cfg.get(channel_key, {}).get(input_type, {}).get("num_components", 0)
    return num_components

def get_selected_feature_extractors(input_types: dict, channel_keys: list) -> dict:
    cfg = _load_active_profile_cfg()
    selected_feature_extractors = {}
    for channel_key in channel_keys:
        input_type = input_types[channel_key]
        selected_feature_extractors[channel_key] = cfg.get(channel_key, {}).get(input_type, {}).get("selected_feature_extractors", [])
    return selected_feature_extractors

def get_default_2D_decay_config() -> tuple:
    cfg = _load_active_profile_cfg()
    default_duration = cfg.get("Decay (2D)", {}).get("duration", 20.0)
    default_time_bins = cfg.get("Decay (2D)", {}).get("time_bins", 1024)
    return default_duration, default_time_bins

def get_default_laser_rate(input_type: str) -> float:
    cfg = _load_active_profile_cfg()
    return cfg.get(input_type, {}).get("laser_rate", 1.0)

def get_decay_input_type() -> str:
    cfg = _load_active_profile_cfg()
    return cfg.get("flim_decay_input_type", "Decay (2D)")

def get_imaging_modality(channel_keys: list) -> dict:
    cfg = _load_active_profile_cfg()
    imaging_modality = {}
    for channel_key in channel_keys:
        imaging_modality[channel_key] = cfg.get(channel_key, {}).get("imaging_modality", "FLIM")
    return imaging_modality

def get_available_feature_extractors(input_type: str) -> list:
    cfg = _load_active_profile_cfg()
    return cfg.get(input_type, {}).get("available_feature_extractors", [])

def get_file_types(input_type: str) -> list:
    cfg = _load_active_profile_cfg()
    return cfg.get(input_type, {}).get("file_types", [])

def get_all_feature_extractors() -> list:
    cfg = _load_active_profile_cfg()
    return cfg.get("all_feature_extractors", [])

def get_categorical_cols() -> list:
    cfg = _load_active_profile_cfg()
    return cfg.get("categorical_cols", [])

def get_derived_features() -> list:
    """Return the active profile's derived-feature definitions.

    Each entry is a dict ``{"name": str, "expression": str, "operands": list[str]}``
    where ``expression`` uses positional aliases (A, B, …) mapped to ``operands``.
    Empty list when the profile defines none — backward-compatible with older
    configs that lack the key.
    """
    return _load_active_profile_cfg().get("derived_features", [])

def set_derived_features(derived_features: list, config_path: Optional[Path] = None) -> None:
    """Persist the active profile's derived-feature list immediately.

    Mirrors the profile create/delete/switch helpers (load → mutate one key →
    save), so an added/deleted derived feature survives a Streamlit rerun without
    waiting for the page-level "Update Configuration" save. Only the
    ``derived_features`` key of the active profile is touched; everything else on
    disk is preserved.
    """
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    current = cfg.get("current_profile", "default")
    cfg.setdefault("profiles", {}).setdefault(current, {})
    cfg["profiles"][current]["derived_features"] = derived_features
    save_config(cfg, config_path)

def get_fit_free_calibration_method(input_type: str) -> tuple[str, float | str]:
    """Return ``(method, standard_lifetime)`` for the given input type.

    ``method`` is the fit-free calibration method (``""`` if unset).
    ``standard_lifetime`` is the reference lifetime (ns) when the method is
    "Fluorescence Lifetime Standard", otherwise ``""``.
    """
    cfg = _load_active_profile_cfg()
    method = cfg.get(input_type, {}).get("fit_free_calibration", "")
    if method == "Fluorescence Lifetime Standard":
        fluorescence_lifetime_standard_lifetime = cfg.get(input_type, {}).get("fluorescence_lifetime_standard_lifetime", "")
        return method, fluorescence_lifetime_standard_lifetime
    else:
        return method, ""

def get_fixed_lifetimes(channel_key: str, input_type: str) -> dict:
    """Return the fixed-lifetime dict for a channel/input_type pair.

    Keys are 't1', 't2', 't3'; values are float (ns) when fixed, or None when free.
    Returns an empty dict when no constraints are stored.
    """
    cfg = _load_active_profile_cfg()
    raw = cfg.get(channel_key, {}).get(input_type, {}).get("fixed_lifetimes", {})
    result = {}
    for key, val in raw.items():
        result[key] = float(val) if (val is not None and float(val) > 0) else None
    return result