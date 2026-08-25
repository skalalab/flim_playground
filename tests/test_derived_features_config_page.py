"""The derived-features builder on the Configuration page (main.py).

Complements the pure-logic tests in test_derived_features.py: this drives the
actual Streamlit page via AppTest to confirm the "Derived features" reveal
checkbox (which carries the section description as its help "?", since
st.expander has no help param) shows/hides the builder and that the builder's
populated path — Template selectbox + operand pickers, and the custom append
buttons — renders without raising.
"""
import copy
import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as config
from src.widgets.derived_features_widgets import DERIVED_FEATURES_HELP

_PAGE = str(Path(__file__).resolve().parents[1] / "main.py")

# Minimal seed: one FLIM channel with a 2-component Lifetime fit selected, so
# predict_feature_columns() yields operands and the builder shows its full UI.
_SEED = {
    "current_profile": "default",
    "profiles": {
        "default": {
            "num_channels": 1,
            "flim_decay_input_type": "Decay (3/4D)",
            "ch1": {
                "imaging_modality": "FLIM",
                "input_type": "Decay (3/4D)",
                "channel_name": "nadh",
                "Decay (3/4D)": {
                    "selected_feature_extractors": ["Lifetime fit"],
                    "num_components": 2,
                },
            },
        }
    },
}


def _write_seed(tmp_path, monkeypatch, derived_features=None):
    seed = copy.deepcopy(_SEED)
    if derived_features is not None:
        seed["profiles"]["default"]["derived_features"] = derived_features
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml.dumps(seed), encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", cfg_path)


def _checkbox(at):
    boxes = [c for c in at.checkbox if c.label == "Derived features"]
    assert boxes, "'Derived features' checkbox not rendered"
    return boxes[0]


def test_checkbox_carries_help_and_is_off_by_default(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)  # no derived features defined
    at = AppTest.from_file(_PAGE).run(timeout=60)
    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"

    cb = _checkbox(at)
    # The description lives on the checkbox's help "?".
    assert cb.help == DERIVED_FEATURES_HELP
    # Off by default when the profile has no derived features -> builder hidden.
    assert cb.value is False
    assert not [s for s in at.selectbox if s.label == "Template"]


def test_ticking_checkbox_reveals_builder(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)
    at = AppTest.from_file(_PAGE).run(timeout=60)

    _checkbox(at).set_value(True).run(timeout=60)
    assert not at.exception, f"reveal raised: {[e.value for e in at.exception]}"

    templates = [s for s in at.selectbox if s.label == "Template"]
    assert templates, "Template selectbox not rendered after tick"
    assert templates[0].value == "Normalized  A / (A + B)"

    operand_boxes = [s for s in at.selectbox if s.label in ("A", "B")]
    assert len(operand_boxes) == 2, "expected operand dropdowns A and B"
    assert "Lifetime fit_nadh: a1" in operand_boxes[0].options


def test_checkbox_defaults_on_when_features_exist(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch, derived_features=[
        {"name": "redox_ratio", "expression": "A/(A+B)",
         "operands": ["Lifetime fit_nadh: a1", "Lifetime fit_nadh: t1"]},
    ])
    at = AppTest.from_file(_PAGE).run(timeout=60)
    assert not at.exception, f"page raised: {[e.value for e in at.exception]}"

    # Existing features must not be hidden: checkbox starts on, feature is shown.
    assert _checkbox(at).value is True
    assert any("redox_ratio" in m.value for m in at.markdown), "existing feature not shown"


def test_custom_mode_renders_buttons_and_formula_box(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)
    at = AppTest.from_file(_PAGE).run(timeout=60)
    _checkbox(at).set_value(True).run(timeout=60)

    templates = [s for s in at.selectbox if s.label == "Template"]
    assert templates, "Template selectbox not rendered after tick"
    # Switch to Custom… — renders the multiselect, append-buttons and formula box.
    templates[0].set_value("Custom…").run(timeout=60)
    assert not at.exception, f"custom mode raised: {[e.value for e in at.exception]}"

    assert any(m.label.startswith("Operands") for m in at.multiselect), "operand multiselect missing"
    assert any(t.label == "Formula" for t in at.text_input), "custom formula box missing"
    # +, -, * are Markdown bullet markers, so their button labels are escaped
    # (else st.button renders them blank); /, (, ) render literally.
    op_labels = {b.label for b in at.button}
    assert {r"\+", r"\-", r"\*", "/", "(", ")"}.issubset(op_labels), f"operator buttons missing: {op_labels}"

    # Clicking an escaped-label button must still append the *raw* operator token.
    star = [b for b in at.button if b.label == r"\*"]
    assert star, "multiply button missing"
    star[0].click().run(timeout=60)
    formula = [t for t in at.text_input if t.label == "Formula"]
    assert formula and "*" in formula[0].value, "clicking * did not append raw '*'"


def _add_button(at):
    # Exact label — the page has other "➕ …" buttons (profile management).
    return [b for b in at.button if b.label == "➕ Add derived feature"][0]


def test_add_clears_custom_builder_inputs(tmp_path, monkeypatch):
    """After adding, the Name, Operands multiselect and Formula box all reset."""
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)  # no derived features yet
    at = AppTest.from_file(_PAGE).run(timeout=60)
    _checkbox(at).set_value(True).run(timeout=60)

    [s for s in at.selectbox if s.label == "Template"][0].set_value("Custom…").run(timeout=60)
    [m for m in at.multiselect if m.label.startswith("Operands")][0].set_value(
        ["Lifetime fit_nadh: a1", "Lifetime fit_nadh: t1"]).run(timeout=60)
    [t for t in at.text_input if t.label == "Name"][0].set_value("myfeat").run(timeout=60)
    [t for t in at.text_input if t.label == "Formula"][0].set_value("A/B").run(timeout=60)

    _add_button(at).click().run(timeout=60)
    assert not at.exception, f"add raised: {[e.value for e in at.exception]}"

    # Saved to config...
    saved = toml.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert [f["name"] for f in saved["profiles"]["default"]["derived_features"]] == ["myfeat"]

    # ...and every builder input is cleared for the next entry.
    assert [t for t in at.text_input if t.label == "Name"][0].value == ""
    assert [m for m in at.multiselect if m.label.startswith("Operands")][0].value == []
    assert [t for t in at.text_input if t.label == "Formula"][0].value == ""


def test_add_bumps_builder_generation(tmp_path, monkeypatch):
    """Reset works by re-keying the inputs (a per-profile generation counter that
    Add bumps), NOT by deleting their session_state — deleting a widget's key does
    not reset it in the live app: Streamlit restores the frontend value on the
    rerun. AppTest has no frontend, so the state-based tests above pass either way
    and can't guard the mechanism. This one does: it fails if the code regresses to
    the delete-the-key approach (which leaves df_gen absent / static).
    """
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)
    at = AppTest.from_file(_PAGE).run(timeout=60)
    _checkbox(at).set_value(True).run(timeout=60)
    assert at.session_state["df_gen_default"] == 0  # fresh builder

    [t for t in at.text_input if t.label == "Name"][0].set_value("g1").run(timeout=60)
    _add_button(at).click().run(timeout=60)
    assert not at.exception, f"add raised: {[e.value for e in at.exception]}"
    assert at.session_state["df_gen_default"] == 1  # bumped -> inputs get new keys

    # A second add bumps again; both features persist independently.
    [t for t in at.text_input if t.label == "Name"][0].set_value("g2").run(timeout=60)
    _add_button(at).click().run(timeout=60)
    assert at.session_state["df_gen_default"] == 2
    saved = toml.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert [f["name"] for f in saved["profiles"]["default"]["derived_features"]] == ["g1", "g2"]


def test_add_resets_template_operand_slots(tmp_path, monkeypatch):
    """Template-mode operand dropdowns (df_op_<profile>_A/B) reset after Add."""
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)
    at = AppTest.from_file(_PAGE).run(timeout=60)
    _checkbox(at).set_value(True).run(timeout=60)  # default template = Normalized

    # Move slot B off its default so a reset is observable.
    b_slot = [s for s in at.selectbox if s.label == "B"][0]
    non_default = next(o for o in b_slot.options if o != b_slot.value)
    b_slot.set_value(non_default).run(timeout=60)
    [t for t in at.text_input if t.label == "Name"][0].set_value("redox").run(timeout=60)

    _add_button(at).click().run(timeout=60)
    assert not at.exception, f"add raised: {[e.value for e in at.exception]}"

    saved = toml.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    feats = saved["profiles"]["default"]["derived_features"]
    assert feats[0]["name"] == "redox" and feats[0]["operands"][1] == non_default

    # Name cleared and slot B fell back to the first option.
    assert [t for t in at.text_input if t.label == "Name"][0].value == ""
    b_after = [s for s in at.selectbox if s.label == "B"][0]
    assert b_after.value == b_after.options[0]


def test_single_operand_formula_is_blocked(tmp_path, monkeypatch):
    """A bare-operand custom formula (e.g. 'A') errors and disables Add."""
    from streamlit.testing.v1 import AppTest

    _write_seed(tmp_path, monkeypatch)
    at = AppTest.from_file(_PAGE).run(timeout=60)
    _checkbox(at).set_value(True).run(timeout=60)

    [s for s in at.selectbox if s.label == "Template"][0].set_value("Custom…").run(timeout=60)
    [m for m in at.multiselect if m.label.startswith("Operands")][0].set_value(
        ["Lifetime fit_nadh: a1"]).run(timeout=60)
    [t for t in at.text_input if t.label == "Name"][0].set_value("dup").run(timeout=60)
    [t for t in at.text_input if t.label == "Formula"][0].set_value("A").run(timeout=60)

    assert any("just duplicates a column" in e.value for e in at.error), "no singleton error"
    assert _add_button(at).disabled, "Add should be disabled for a lone operand"
