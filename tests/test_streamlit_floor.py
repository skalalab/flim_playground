"""Streamlit 1.67 stops a widget from rerunning the app when an interaction leaves
its value unchanged (streamlit/streamlit#17822). Before it, opening a multiselect
such as Color by and closing it without a change reran the whole page, rebuilding
the figure (about 0.9 s at 200k cells). AppTest cannot close a dropdown, so these
checks pin the version: the floor for installs without the lock, and the lock for
the release builds, which run ``uv sync --locked``.
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = (1, 67)


def _version(text):
    return tuple(int(part) for part in text.split(".")[:2])


def test_pyproject_requires_streamlit_that_skips_unchanged_widget_reruns():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'"streamlit>=(\d+\.\d+)', pyproject)
    assert floor and _version(floor.group(1)) >= FLOOR, floor and floor.group(0)


def test_lock_pins_streamlit_that_skips_unchanged_widget_reruns():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = next(p["version"] for p in lock["package"] if p["name"] == "streamlit")
    assert _version(locked) >= FLOOR, locked
