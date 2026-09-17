"""The Configuration page seeds a usable default profile when config.toml is absent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

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


def _profile_selector(at):
    boxes = [s for s in at.selectbox if s.label == "Profile"]
    assert len(boxes) == 1, [s.key for s in boxes]
    return boxes[0]


def _create_profile(at, name):
    at.text_input(key="new_extraction_profile_name").set_value(name)
    create = next(b for b in at.button if "Create" in str(b.label))
    return create.click().run(timeout=60)


def test_creating_a_profile_switches_the_selector_to_it(tmp_path, monkeypatch):
    """The selector must show the new profile right away.

    Streamlit identifies a keyed selectbox by its key alone, so a browser that
    already rendered the selector keeps its old choice unless the widget comes
    back under a fresh key. AppTest has no browser, so the test guards the
    re-key rather than the displayed value.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "_CONFIG_PATH", tmp_path / "config.toml")
    at = AppTest.from_file(_PAGE).run(timeout=60)
    before = _profile_selector(at)
    assert before.value == "default"

    at = _create_profile(at, "experiment-B")

    assert not at.exception, [e.value for e in at.exception]
    after = _profile_selector(at)
    assert after.value == "experiment-B"
    assert after.key != before.key, "selector kept its identity, so a browser would restore the old choice"
    assert config.get_current_profile_name() == "experiment-B"

    # A plain rerun keeps the new profile selected.
    at = at.run(timeout=60)
    assert _profile_selector(at).value == "experiment-B"
    assert config.get_current_profile_name() == "experiment-B"


def test_deleting_a_profile_switches_the_selector_to_the_remaining_one(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "_CONFIG_PATH", tmp_path / "config.toml")
    at = AppTest.from_file(_PAGE).run(timeout=60)
    at = _create_profile(at, "experiment-B")
    before = _profile_selector(at)
    assert before.value == "experiment-B"

    at = at.button(key="delete_extraction_profile").click().run(timeout=60)

    assert not at.exception, [e.value for e in at.exception]
    after = _profile_selector(at)
    assert after.value == "default"
    assert after.key != before.key, "selector kept its identity, so a browser would restore the deleted choice"
    assert config.list_profiles() == ["default"]
    assert config.get_current_profile_name() == "default"
