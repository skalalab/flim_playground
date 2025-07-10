import toml
from pathlib import Path
from collections import defaultdict

# Absolute path to the project-level config file (../config.toml)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def load_config() -> dict:
    """Load and return the TOML configuration as a dict.

    Returns an empty dict if the file is missing or unparsable so that the
    calling code can supply sensible fall-backs.
    """
    try:
        return toml.load(_CONFIG_PATH)
    except (FileNotFoundError, toml.TomlDecodeError):
        return {}


def save_config(cfg: dict) -> None:
    """Persist *cfg* to disk, overwriting the previous *config.toml*."""
    # Ensure the parent directory exists (helpful when running tests, etc.)
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CONFIG_PATH.open("w", encoding="utf-8") as fh:
        toml.dump(cfg, fh)


def get_cell_id_column_name() -> str:
    cfg = load_config()
    cell_id_column_name = cfg.get("unique_cell_id_col", "cell_id")
    return cell_id_column_name

def get_file_suffixes(channel_name: str, input_type: str) -> dict:
    cfg = load_config()
    channel_feature_types = cfg.get("feature_types", {}).get(channel_name, [])
    suffixes = {}
    for feature_type in channel_feature_types:
        file_type_suffixes = cfg.get("inputSuffixes", {}).get(channel_name, {}).get(feature_type, {}).get(input_type, {})
        for file_type, suffix in file_type_suffixes.items():
            if file_type not in suffixes:
                suffixes[file_type] = suffix
    return suffixes
    

def get_available_input_types() -> list:
    cfg = load_config()
    available_input_types = cfg.get("available_input_types", [])
    preferred_input_type = cfg.get("preferred_input_type", None)
    if preferred_input_type is not None:
        preferred_input_type_index = available_input_types.index(preferred_input_type)
    else:
        preferred_input_type_index = 0
    return available_input_types, preferred_input_type_index

def get_channel_names() -> dict:
    cfg = load_config()
    num_channels = cfg.get("num_channels", 0)
    all_channel_names = cfg.get("channel_names", {})
    channel_names = {}
    for i, (channel_key, channel_name) in enumerate(all_channel_names.items()):
        if i < num_channels:
            channel_names[channel_key] = channel_name
    return channel_names

def get_spc_output_suffix() -> dict:
    cfg = load_config()
    spc_output_suffix = cfg.get("spc_output_suffix", {})
    return spc_output_suffix

def get_num_components(channel_names: list) -> dict:
    cfg = load_config()
    num_components = {}
    for channel_name in channel_names:
        num_components[channel_name] = cfg.get("num_components", {}).get(channel_name, 0)
    return num_components

def get_feature_types(channel_names: list) -> dict:
    cfg = load_config()
    feature_types = {}
    for channel_name in channel_names:
        feature_types[channel_name] = cfg.get("feature_types", {}).get(channel_name, [])
    return feature_types