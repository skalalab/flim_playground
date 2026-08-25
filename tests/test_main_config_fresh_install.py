"""The Configuration page (main.py) must not crash on a fresh install.

When FLIM Playground launches for the very first time with no config.toml
anywhere -- e.g. a bundled-app first run, or a build that does not ship a seed
config -- ``load_config()`` returns ``{}`` and ``list_profiles()`` returns
``[]``. The profile selectbox then renders with empty options and returns
``None``, which used to call ``set_current_profile(None)`` ->
``toml.dump`` ``KeyError: 'None'``, crashing the page *before* the
default-seeding logic (main.py:108+) could ever run.

The page should instead fall back to the ``default`` profile and seed app
defaults so the user lands on a usable Configuration page with no bundled
config required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as config

_PAGE = str(Path(__file__).resolve().parents[1] / "main.py")


def test_fresh_install_no_config_does_not_crash(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    # Fresh install: point the config path at a file that does not exist
    # (nothing bundled, nothing previously saved).
    missing = tmp_path / "config.toml"
    assert not missing.exists()
    monkeypatch.setattr(config, "_CONFIG_PATH", missing)

    at = AppTest.from_file(_PAGE).run(timeout=60)

    # The page must not crash on the empty-profiles selectbox path.
    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"

    # It should fall back to a real ``default`` profile rather than an empty
    # selectbox returning None.
    profile_boxes = [s for s in at.selectbox if s.label == "Profile"]
    assert profile_boxes, "Profile selectbox not rendered"
    assert profile_boxes[0].value == "default", (
        f"expected fallback to 'default' profile, got {profile_boxes[0].value!r}"
    )
