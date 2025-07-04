import toml
from pathlib import Path


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


def get_file_suffix_default() -> dict:
    """Return a flat dict mapping each logical input file to its suffix.

    This reshapes the hierarchical TOML structure into the legacy flat
    `file_suffix_default` layout that the rest of the codebase expects.
    """
    cfg = load_config()

    file_suffix_tbl = cfg.get("file_suffix", {})
    channel_names = cfg.get("channel_name", {})

    file_suffix_default: dict[str, str] = {}
    file_suffix_default["mask"] = file_suffix_tbl.get("mask", "_mask.tiff")

    channels = ["blue", "green", "red"]
    input_files = ["irf", "a1", "histogram", "decay"]

    for channel in channels:
        channel_name = channel_names.get(channel, channel)
        for input_file in input_files:
            key = f"{channel}_{input_file}"
            file_suffix_default[f"{channel_name} {input_file}"] = file_suffix_tbl.get(key, "")

    return file_suffix_default






