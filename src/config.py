import toml
import sys
from pathlib import Path
from typing import Optional

def _get_config_path() -> Path:
    """Get the config file path, handling both development and bundled app scenarios."""
    # Check if running as a PyInstaller bundle
    if getattr(sys, '_MEIPASS', None):
        # Running as bundled app - save config to a persistent location
        # Use the directory where the executable is located
        exe_dir = Path(sys.executable).parent
        config_path = exe_dir / "config.toml"
        
        # If config doesn't exist in exe directory, try to copy from bundle
        if not config_path.exists():
            bundled_config = Path(sys._MEIPASS) / "config.toml"
            if bundled_config.exists():
                import shutil
                shutil.copy(bundled_config, config_path)
        
        return config_path
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

def get_unique_cell_id_col() -> str:
    cfg = load_config()
    return cfg.get("unique_cell_id_col", "cell_id")

def get_fov_name_col() -> str:
    cfg = load_config()
    return cfg.get("fov_name_col", "image_name")

def get_input_types(channel_keys: list) -> dict:
    cfg = load_config()
    input_types = {}
    for channel_key in channel_keys:
        input_types[channel_key] = cfg.get(channel_key, {}).get("input_type", None)
    return input_types

def get_default_file_suffixes(channel_key: str, input_type: str, selected_feature_extractors: list) -> dict:
    cfg = load_config()
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
                                        ("prefitted" in input_type and not "Lifetime fit free" in selected_feature_extractors)):
            continue
        if file_type == "IRF" and ("Lifetime fit" not in selected_feature_extractors or "prefitted" in input_type) and "Lifetime fit free" in selected_feature_extractors and fit_free_calibration == "Fluorescence Lifetime Standard":
            continue
        filtered_file_suffixes[file_type] = file_suffixes[file_type]
    return filtered_file_suffixes

def get_channel_names() -> dict:
    cfg = load_config()
    num_channels = cfg.get("num_channels", 0)
    channel_names = {}
    for i in range(num_channels):
        channel_key = f"ch{i+1}"
        channel_names[channel_key] = cfg.get(channel_key, {}).get("channel_name", channel_key)
    return channel_names

def get_spc_output_suffix() -> dict:
    cfg = load_config()
    spc_output_suffix = cfg.get("spc_output_suffix", {})
    return spc_output_suffix

def get_num_components(input_types: dict, channel_keys: list) -> dict:
    cfg = load_config()
    num_components = {}
    for channel_key in channel_keys:
        input_type = input_types[channel_key]
        num_components[channel_key] = cfg.get(channel_key, {}).get(input_type, {}).get("num_components", 0)
    return num_components

def get_selected_feature_extractors(input_types: dict, channel_keys: list) -> dict:
    cfg = load_config()
    selected_feature_extractors = {}
    for channel_key in channel_keys:
        input_type = input_types[channel_key]
        selected_feature_extractors[channel_key] = cfg.get(channel_key, {}).get(input_type, {}).get("selected_feature_extractors", [])
    return selected_feature_extractors

def get_default_2D_decay_config() -> tuple:
    cfg = load_config()
    default_duration = cfg.get("Decay (2D)", {}).get("duration", 20.0)
    default_time_bins = cfg.get("Decay (2D)", {}).get("time_bins", 1024)
    return default_duration, default_time_bins

def get_default_laser_rate(input_type: str) -> float:
    cfg = load_config()
    return cfg.get(input_type, {}).get("laser_rate", 1.0)

def get_decay_input_type() -> str:
    cfg = load_config()
    return cfg.get("flim_decay_input_type", "Decay (2D)")

def get_imaging_modality(channel_keys: list) -> dict:
    cfg = load_config()
    imaging_modality = {}
    for channel_key in channel_keys:
        imaging_modality[channel_key] = cfg.get(channel_key, {}).get("imaging_modality", "FLIM")
    return imaging_modality

def get_available_feature_extractors(input_type: str) -> list:
    cfg = load_config()
    return cfg.get(input_type, {}).get("available_feature_extractors", [])

def get_file_types(input_type: str) -> list:
    cfg = load_config()
    return cfg.get(input_type, {}).get("file_types", [])

def get_all_feature_extractors() -> list:
    cfg = load_config()
    return cfg.get("all_feature_extractors", [])

def get_categorical_cols() -> list:
    cfg = load_config()
    return cfg.get("categorical_cols", [])

def get_fit_free_calibration_method(input_type: str) -> str:
    cfg = load_config()
    method = cfg.get(input_type, {}).get("fit_free_calibration", "")
    if method == "Fluorescence Lifetime Standard":
        fluorescence_lifetime_standard_lifetime = cfg.get(input_type, {}).get("fluorescence_lifetime_standard_lifetime", "")
        return method, fluorescence_lifetime_standard_lifetime
    else:
        return method, "", ""