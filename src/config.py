import os
import sys
from pathlib import Path

import toml


def get_persistent_dir() -> Path:
    """Return a writable config directory outside the bundled app payload.

    Use the directory beside a macOS .app, the XDG config directory under
    flim-playground on Linux, and the executable's directory otherwise.
    """
    exe = Path(sys.executable)
    if exe.parent.name == "MacOS" and exe.parent.parent.name == "Contents":
        return exe.parent.parent.parent.parent  # directory containing the .app
    if sys.platform.startswith("linux"):
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
        return Path(base) / "flim-playground"
    return exe.parent  # Windows: beside the .exe


def _get_config_path() -> Path:
    """Use persistent storage for bundles and the project root for development."""
    if getattr(sys, '_MEIPASS', None):
        # main.py seeds defaults when the persistent config is missing.
        return get_persistent_dir() / "config.toml"
    else:
        return Path(__file__).resolve().parent.parent / "config.toml"

# Resolved once for the current runtime.
_CONFIG_PATH = _get_config_path()

def load_config(config_path: Path | None = None) -> dict:
    """Load and return the TOML configuration as a dict.

    Returns an empty dict if the file is missing or unparsable so that the
    calling code can supply sensible fall-backs.
    """
    path_to_load = _CONFIG_PATH if config_path is None else config_path
    try:
        return toml.load(path_to_load)
    except (FileNotFoundError, toml.TomlDecodeError):
        return {}

def save_config(cfg: dict, config_path: Path | None = None) -> None:
    """Persist *cfg* to disk, overwriting the previous config file."""
    path_to_save = _CONFIG_PATH if config_path is None else config_path
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    with path_to_save.open("w", encoding="utf-8") as fh:
        toml.dump(cfg, fh)

def name_survives_round_trip(name: str) -> bool:
    """Whether the TOML library preserves *name* as a table key through a save/load.

    Profile lookup requires the stored key to match the typed name exactly.
    Testing the library directly keeps validation aligned with its escaping rules.
    """
    try:
        return list(toml.loads(toml.dumps({name: {}}))) == [name]
    except (toml.TomlDecodeError, TypeError, ValueError):
        return False

# Spell out characters that would be hard to see in an error message.
_UNSTORABLE_CHAR_NAMES = {"\\": "a backslash", '"': "a double quote", "\t": "a tab",
                          "\n": "a line break", "\r": "a carriage return"}


def unstorable_name_error(name: str) -> str:
    """Explain a profile name's TOML round-trip failure, or return "".

    Shared by extraction and analysis profiles. Test individual characters with
    the same round trip to identify the cause when possible.
    """
    if name_survives_round_trip(name):
        return ""
    named = list(dict.fromkeys(
        _UNSTORABLE_CHAR_NAMES.get(ch) or (f"{ch!r}" if ch.isprintable()
                                           else "a control character")
        for ch in dict.fromkeys(name) if not name_survives_round_trip(ch)))
    if len(named) > 1:
        culprit = ", ".join(named[:-1]) + " or " + named[-1]
    else:
        culprit = named[0] if named else "that character"
    return (f"A profile name cannot contain {culprit}: the config file stores it as a "
            "section header and reads that character back escaped, so the profile could "
            "not be found again under the name you typed and the next save would write a "
            "second one. Pick another.")

def get_config_mtime(config_path: Path | None = None) -> float:
    """Return the config file's last-modified time, or ``0.0`` if it is missing.

    Open tabs can poll this value to detect configuration edits in other sessions.
    """
    path = _CONFIG_PATH if config_path is None else config_path
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0

# Extraction configuration format:
#   current_profile = "default"
#   [profiles.<name>]   # complete extraction settings
#
# Read current_profile from disk to keep this module Streamlit-free. main.py
# persists profile switches, creation, and deletion before rerunning.

def _migrate_extraction_config_to_profiles(cfg: dict) -> dict:
    """Wrap a flat extraction config under ``profiles.default``.

    Idempotent: a config that already has a ``profiles`` key is returned
    unchanged. An empty dict (missing/unparsable file) is left untouched so the
    caller can seed defaults. The new structure is built in memory and returned;
    the input is not mutated.
    """
    if "profiles" in cfg:
        return cfg
    if not cfg:
        return cfg
    # current_profile is metadata, not an extraction setting.
    default_profile = {k: v for k, v in cfg.items() if k != "current_profile"}
    return {"current_profile": "default", "profiles": {"default": default_profile}}

def _load_active_profile_cfg(config_path: Path | None = None) -> dict:
    """Return the active profile's extraction settings, or {} if absent."""
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    current = cfg.get("current_profile", "default")
    return cfg.get("profiles", {}).get(current, {})

def get_current_profile_name(config_path: Path | None = None) -> str:
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    return cfg.get("current_profile", "default")

def list_profiles(config_path: Path | None = None) -> list:
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    return list(cfg.get("profiles", {}).keys())

def set_current_profile(name: str, config_path: Path | None = None) -> None:
    """Switch the active profile, creating it (empty) if it does not exist."""
    cfg = _migrate_extraction_config_to_profiles(load_config(config_path))
    cfg.setdefault("profiles", {}).setdefault(name, {})
    cfg["current_profile"] = name
    save_config(cfg, config_path)

def create_profile(name: str, config_path: Path | None = None) -> None:
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

def delete_profile(name: str, config_path: Path | None = None) -> None:
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
    for file_type in file_suffixes:
        # Prefitted lifetime-only extraction does not need the raw decay.
        if file_type == "Decay" and "prefitted" in input_type and len(selected_feature_extractors) == 1 and "Lifetime fit" in selected_feature_extractors:
            continue
        if file_type == "Fluorescence Lifetime Standard" and not (
            "Lifetime fit free" in selected_feature_extractors and fit_free_calibration == "Fluorescence Lifetime Standard"
        ):
            continue
        # SPCImage t1 is used only by the lifetime-fit extractor.
        if file_type == "SPCImage t1" and "Lifetime fit" not in selected_feature_extractors:
            continue
        # An IRF is needed only for lifetime extraction that uses it for calibration.
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
    Return an empty list when the profile defines none.
    """
    return _load_active_profile_cfg().get("derived_features", [])

def set_derived_features(derived_features: list, config_path: Path | None = None) -> None:
    """Persist the active profile's derived-feature list immediately.

    Changes survive reruns without waiting for "Update Configuration". Update
    only the active profile's ``derived_features`` key, preserving other settings.
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
