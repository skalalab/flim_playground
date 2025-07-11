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


def get_unique_cell_id_col() -> str:
    cfg = load_config()
    return cfg.get("unique_cell_id_col", "cell_id")

def get_image_name_col() -> str:
    cfg = load_config()
    return cfg.get("image_name_col", "image_name")

def get_file_suffixes(channel_name: str, input_type: str) -> dict:
    cfg = load_config()
    suffixes = {}
    file_type_suffixes = cfg.get("inputSuffixes", {}).get(channel_name, {}).get(input_type, {})
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

def get_feature_extractors(channel_names: list) -> dict:
    cfg = load_config()
    feature_extractors = {}
    for channel_name in channel_names:
        feature_extractors[channel_name] = cfg.get("feature_extractors", {}).get(channel_name, {})
    return feature_extractors