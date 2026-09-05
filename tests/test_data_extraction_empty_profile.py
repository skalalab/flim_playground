"""An empty extraction profile shows guidance to configure channels and stops.
Profiles with channels render the extraction workflow normally.
"""
import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as config

_PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_extraction.py")


def _point_config_at(tmp_path, monkeypatch, cfg):
    p = tmp_path / "config.toml"
    p.write_text(toml.dumps(cfg))
    monkeypatch.setattr(config, "_CONFIG_PATH", p)
    return p


def test_empty_active_profile_does_not_crash(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _point_config_at(tmp_path, monkeypatch, {
        "current_profile": "blank",
        "profiles": {"blank": {}, "default": {"num_channels": 1}},
    })

    at = AppTest.from_file(_PAGE).run(timeout=60)

    # Render configuration guidance without an exception.
    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"
    warnings = " ".join(w.value.lower() for w in at.warning)
    assert "channel" in warnings or "configuration" in warnings, (
        f"expected a no-channels warning, got warnings={[w.value for w in at.warning]}"
    )


def test_configured_profile_still_renders(tmp_path, monkeypatch):
    """A profile with channels must reach the normal page (no false positive)."""
    from streamlit.testing.v1 import AppTest

    _point_config_at(tmp_path, monkeypatch, {
        "current_profile": "exp",
        "profiles": {
            "exp": {
                "num_channels": 1,
                "flim_decay_input_type": "Decay (3/4D)",
                "ch1": {"channel_name": "NADH", "input_type": "Decay (3/4D)"},
            },
        },
    })

    at = AppTest.from_file(_PAGE).run(timeout=60)

    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"
    # The normal page renders the step radio; the no-channels guard must NOT fire.
    warnings = " ".join(w.value.lower() for w in at.warning)
    assert "no channels" not in warnings


# Each extraction step renders, with an explicit guard for an empty FOV channel selection.

_STEPS = [
    "FOV Metadata Extraction",
    "Numeric Feature Extraction (fitting, phasor, etc.)",
    "Categorical Feature Extraction (e.g. treatment)",
]


def _configured_cfg():
    return {
        "current_profile": "exp",
        "profiles": {
            "exp": {
                "num_channels": 1,
                "flim_decay_input_type": "Decay (3/4D)",
                "ch1": {"channel_name": "NADH", "input_type": "Decay (3/4D)"},
            },
        },
    }


def _step_radio(at):
    for r in at.radio:
        if r.label == "Select a step to perform":
            return r
    raise AssertionError(f"step radio not found; radios={[r.label for r in at.radio]}")


def test_step_radio_lists_exactly_the_three_steps(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _point_config_at(tmp_path, monkeypatch, _configured_cfg())
    at = AppTest.from_file(_PAGE).run(timeout=60)

    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"
    assert _step_radio(at).options == _STEPS


def test_every_step_renders_without_exception(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _point_config_at(tmp_path, monkeypatch, _configured_cfg())
    at = AppTest.from_file(_PAGE).run(timeout=60)
    assert not at.exception, f"initial render raised: {[e.value for e in at.exception]}"

    for step in _STEPS:
        _step_radio(at).set_value(step).run(timeout=60)
        assert not at.exception, f"step {step!r} raised: {[e.value for e in at.exception]}"


def test_fov_step_errors_when_no_channel_selected(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _point_config_at(tmp_path, monkeypatch, _configured_cfg())
    at = AppTest.from_file(_PAGE).run(timeout=60)
    assert not at.exception

    # Default step is FOV; the single channel is checked by default -> uncheck it.
    channel_boxes = [c for c in at.checkbox if c.label.startswith("has ")]
    assert channel_boxes, f"no channel checkbox; checkboxes={[c.label for c in at.checkbox]}"
    channel_boxes[0].set_value(False).run(timeout=60)

    assert not at.exception
    errors = " ".join(e.value.lower() for e in at.error)
    assert "at least one" in errors, f"expected the no-channel error, got {[e.value for e in at.error]}"
