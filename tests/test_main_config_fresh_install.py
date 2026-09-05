"""The Configuration page seeds a usable default profile when config.toml is absent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as config

_PAGE = str(Path(__file__).resolve().parents[1] / "main.py")


def test_fresh_install_no_config_does_not_crash(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    missing = tmp_path / "config.toml"
    assert not missing.exists()
    monkeypatch.setattr(config, "_CONFIG_PATH", missing)

    at = AppTest.from_file(_PAGE).run(timeout=60)

    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"

    profile_boxes = [s for s in at.selectbox if s.label == "Profile"]
    assert profile_boxes, "Profile selectbox not rendered"
    assert profile_boxes[0].value == "default", (
        f"expected fallback to 'default' profile, got {profile_boxes[0].value!r}"
    )


def test_a_profile_name_the_config_file_cannot_store_is_refused(tmp_path, monkeypatch):
    """Reject names that the TOML serializer cannot read back unchanged."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "_CONFIG_PATH", tmp_path / "config.toml")
    at = AppTest.from_file(_PAGE).run(timeout=60)

    boxes = [w for w in at.text_input if w.key == "new_extraction_profile_name"]
    assert boxes, [w.key for w in at.text_input]
    boxes[0].set_value("run\\2026").run(timeout=60)
    create = [b for b in at.button if "Create" in str(b.label)]
    assert create, [str(b.label) for b in at.button]
    at = create[0].click().run(timeout=60)

    assert any("cannot contain a backslash" in str(e.value) for e in at.error), \
        [str(e.value) for e in at.error]
    assert not any("2026" in name for name in config.list_profiles()), config.list_profiles()
