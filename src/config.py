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








