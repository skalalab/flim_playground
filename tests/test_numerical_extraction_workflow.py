"""Exercise the preparation/calibration/extraction transitions in the real page."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
import tifffile
import toml

from src import config
from src.widgets import lifetime_widgets, metadata_widgets, numeric_extraction_widgets


PAGE = str(Path(__file__).resolve().parents[1] / "pages/data_extraction.py")
CONFIRM = "Confirm calibration for each channel"


def button(app, label):
    return next(b for b in app.button if b.label == label)


def run(app):
    # Mirror the browser after widgets are removed by a Streamlit rerun.
    from streamlit.proto.WidgetStates_pb2 import WidgetStates
    from streamlit.testing.v1.element_tree import Widget
    states = WidgetStates()
    for node in app:
        if isinstance(node, Widget) and node.id in app.session_state:
            states.widgets.append(node._widget_state)
    app._run(states, timeout=30)
    assert not app.exception, [e.value for e in app.exception]
    return app


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("fov1", "fov2"):
        np.savetxt(folder / f"{name}_decay.csv", np.ones((2, 16)), delimiter=",")
    np.savetxt(folder / "irf.csv", np.ones(16), delimiter=",")
    cfg = {
        "num_channels": 1, "flim_decay_input_type": "Decay (2D)",
        "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
        "Decay (2D)": {"available_feature_extractors": ["Lifetime fit"],
                       "file_types": ["Decay", "IRF"], "duration": 12.5, "time_bins": 16},
        "ch1": {"channel_name": "NADH", "imaging_modality": "FLIM", "input_type": "Decay (2D)",
                "Decay (2D)": {"selected_feature_extractors": ["Lifetime fit"],
                               "num_components": 2,
                               "input_suffixes": {"Decay": "_decay.csv", "IRF": "irf.csv"}}},
    }
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(cfg))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    metadata_widgets.clear_folder_scan_caches()
    state = {"shift": 1.0, "calls": []}
    monkeypatch.setattr(lifetime_widgets, "choose_shift_widget", lambda *a, **kw: ("", state["shift"]))

    def extract(rows, settings):
        state["calls"].append((rows.copy(), settings["fitting_mode"]))
        return pd.DataFrame({"image_name": ["fov1"], "feature": [42.0]},
                            index=pd.Index(["fov1_1"], name="cell_id"))

    monkeypatch.setattr(numeric_extraction_widgets, "fov_extraction_widget", extract)
    app = AppTest.from_file(PAGE).run(timeout=30)
    assert not app.exception
    app.text_input(key="fov_metadata_folder_path").set_value(str(folder)).run(timeout=30)
    assert not app.exception
    return app, folder, state, path


def prepare_and_calibrate(app):
    button(app, "Start calibration").click().run(timeout=30)
    assert not app.exception
    button(app, "Optimize for Shifts").click().run(timeout=30)
    run(app)
    button(app, CONFIRM).click().run(timeout=30)
    run(app)
    assert not app.error, [e.value for e in app.error]
    assert app.selectbox(key="fitting_mode_update").value == "Local"


def test_two_steps_source_folders_only_and_no_manual_metadata_export(workflow, monkeypatch):
    from src import config_watch
    app, folder, _, _ = workflow
    runs = []
    monkeypatch.setattr(config_watch, "notify_on_config_change", lambda: runs.append(1))
    assert app.radio[0].options == ["**Numerical** (e.g. lifetime, morphology)", "**Categorical** (e.g. treatment, day)"]
    assert not app.get("file_uploader")
    assert not list(folder.glob("fov_metadata_*.csv"))
    assert not any("metadata" in b.label.lower() for b in app.button)
    button(app, "Start calibration").click().run(timeout=30)
    assert len(list(folder.glob("fov_metadata_*.csv"))) == 1
    # Preparation reruns at once so the browser drops the Prepare button within
    # the same interaction. AppTest merges both runs' elements, so count runs.
    assert len(runs) == 2
    run(app)
    run(app)
    assert len(list(folder.glob("fov_metadata_*.csv"))) == 1


def test_mode_edit_and_recalibration_save_same_csv_and_wait_for_start(workflow):
    app, folder, state, _ = workflow
    prepare_and_calibrate(app)
    path = next(folder.glob("fov_metadata_*.csv"))
    assert pd.read_csv(path)["NADH_shift"].eq(1).all()
    button(app, "Start extraction").click().run(timeout=30)
    run(app)
    assert len(state["calls"]) == 1
    exported = next(folder.glob("single_cell_features_*.csv"))
    original_features = exported.read_bytes()
    app.selectbox(key="fitting_mode_update").set_value("Hybrid").run(timeout=30)
    assert not app.exception
    assert len(state["calls"]) == 1
    assert app.session_state["prepared_extraction"].features is None
    saved = pd.read_csv(path)
    assert saved["NADH_shift"].eq(1).all()
    assert saved["fitting_mode"].eq("Hybrid").all()
    run(app)
    assert app.selectbox(key="fitting_mode_update").value == "Hybrid"
    button(app, "Go back and find shift").click().run(timeout=30)
    run(app)
    assert not any(b.label == "Start extraction" for b in app.button)
    assert path.read_bytes() == saved.to_csv(index=False).encode()
    state["shift"] = 2.5
    app.number_input(key="NADH_start").set_value(2).run(timeout=30)
    app.selectbox(key="fitting_metric").set_value("WLS").run(timeout=30)
    button(app, CONFIRM).click().run(timeout=30)
    run(app)
    assert app.selectbox(key="fitting_mode_update").value == "Hybrid"
    saved = pd.read_csv(path)
    assert saved["NADH_shift"].eq(2.5).all()
    assert saved["NADH_start"].eq(2).all()
    assert saved["fitting_algo"].eq("WLS").all()
    assert len(state["calls"]) == 1
    assert len(list(folder.glob("fov_metadata_*.csv"))) == 1
    assert exported.read_bytes() == original_features
    button(app, "Go back and find shift").click().run(timeout=30)
    run(app)
    assert app.number_input(key="NADH_start").value == 2
    assert app.selectbox(key="fitting_metric").value == "WLS"


def test_failed_mode_save_blocks_extraction_until_retry(workflow, monkeypatch):
    from src import extraction_session
    app, folder, state, _ = workflow
    prepare_and_calibrate(app)
    path = next(folder.glob("fov_metadata_*.csv"))
    before = path.read_bytes()

    def fail(*args):
        raise PermissionError("locked file")

    with monkeypatch.context() as patch:
        patch.setattr(extraction_session.os, "replace", fail)
        app.selectbox(key="fitting_mode_update").set_value("Hybrid").run(timeout=30)
        assert not app.exception
        assert any("locked file" in e.value for e in app.error)
        assert button(app, "Start extraction").disabled
        assert path.read_bytes() == before
        assert not state["calls"]
    button(app, "Retry saving metadata").click().run(timeout=30)
    run(app)
    assert not button(app, "Start extraction").disabled
    assert pd.read_csv(path)["fitting_mode"].eq("Hybrid").all()
    assert not state["calls"]


def test_step_switch_keeps_prepared_mode_and_exports_without_repeating_extraction(workflow):
    app, folder, state, _ = workflow
    prepare_and_calibrate(app)
    app.selectbox(key="fitting_mode_update").set_value("Hybrid").run(timeout=30)
    button(app, "Start extraction").click().run(timeout=30)
    prepared = app.session_state["prepared_extraction"]
    app.radio[0].set_value("**Categorical** (e.g. treatment, day)").run(timeout=30)
    assert not app.exception
    app.radio[0].set_value("**Numerical** (e.g. lifetime, morphology)").run(timeout=30)
    assert not app.exception
    assert app.session_state["prepared_extraction"] is prepared
    assert app.selectbox(key="fitting_mode_update").value == "Hybrid"
    assert len(state["calls"]) == 1
    assert len(list(folder.glob("single_cell_features_*.csv"))) == 1


@pytest.mark.parametrize("stage", ["preparation", "calibration", "extraction"])
def test_each_automatic_metadata_save_failure_preserves_edits_and_blocks_run(workflow, monkeypatch, stage):
    from src import extraction_session
    app, folder, state, _ = workflow
    previous = None
    if stage == "calibration":
        button(app, "Start calibration").click().run(timeout=30)
        button(app, "Optimize for Shifts").click().run(timeout=30)
        previous = next(folder.glob("fov_metadata_*.csv")).read_bytes()
    elif stage == "extraction":
        prepare_and_calibrate(app)
        previous = next(folder.glob("fov_metadata_*.csv")).read_bytes()

    def fail(*args):
        raise PermissionError("locked file")

    label = {"preparation": "Start calibration", "calibration": CONFIRM,
             "extraction": "Start extraction"}[stage]
    with monkeypatch.context() as patch:
        patch.setattr(extraction_session.os, "replace", fail)
        button(app, label).click().run(timeout=30)
        run(app)
        prepared = app.session_state["prepared_extraction"]
        assert prepared is not None
        assert any("locked file" in e.value for e in app.error)
        assert not state["calls"]
        if previous is None:
            assert not prepared.metadata_path.exists()
        else:
            assert prepared.metadata_path.read_bytes() == previous
        if stage == "calibration":
            assert prepared.metadata_df["NADH_shift"].eq(1).all()
        if stage != "preparation":
            assert button(app, "Start extraction").disabled
    button(app, "Retry saving metadata").click().run(timeout=30)
    run(app)
    assert prepared.metadata_path.exists()
    assert len(list(folder.glob("fov_metadata_*.csv"))) == 1
    assert not prepared.metadata_error
    assert not state["calls"]


@pytest.mark.parametrize("change", ["folder", "suffix", "channel", "duration", "rescan", "profile"])
def test_source_setup_edits_invalidate_preparation(workflow, change):
    app, folder, state, cfg_path = workflow
    prepare_and_calibrate(app)
    button(app, "Start extraction").click().run(timeout=30)
    if change == "folder":
        app.text_input(key="fov_metadata_folder_path").set_value("")
    elif change == "suffix":
        app.text_input(key="NADH_Decay (2D)_Decay_suffix").set_value("missing.csv")
    elif change == "channel":
        app.checkbox(key="has_channel_ch1").set_value(False)
    elif change == "duration":
        app.number_input(key="2D_decay_duration").set_value(10.0)
    elif change == "rescan":
        button(app, "Rescan folder").click()
    else:
        cfg = toml.load(cfg_path)
        profile = cfg["profiles"][cfg["current_profile"]] if "profiles" in cfg else cfg
        profile["ch1"]["Decay (2D)"]["num_components"] = 3
        cfg_path.write_text(toml.dumps(cfg))
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["prepared_extraction"] is None
    assert not any(b.label == "Start extraction" for b in app.button)
    assert len(state["calls"]) == 1


def no_calibration_app(tmp_path, monkeypatch, kind):
    """Open the page on a folder whose channels need no shift calibration."""
    folder = tmp_path / "images"
    folder.mkdir()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[4:7, 4:7] = 2
    tifffile.imwrite(folder / "sample_mask.tif", mask)
    if kind == "intensity":
        input_type = "Intensity (2D)"
        extractors = ["Intensity morphology"]
        suffixes = {"Intensity (2D)": "_intensity.tif", "Mask": "_mask.tif"}
        tifffile.imwrite(folder / "sample_intensity.tif", np.full((8, 8), 100, dtype=np.uint16))
    else:
        input_type = "Decay (3/4D) pixel-prefitted"
        extractors = ["Lifetime fit"]
        suffixes = {"SPCImage t1": "_t1.asc", "Mask": "_mask.tif"}
        np.savetxt(folder / "sample_t1.asc", np.full((8, 8), 1200.0))
    cfg = {
        "num_channels": 1, "flim_decay_input_type": input_type,
        "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
        input_type: {"available_feature_extractors": extractors, "file_types": list(suffixes)},
        "ch1": {"channel_name": "NADH", "input_type": input_type,
                "imaging_modality": "Intensity-only" if kind == "intensity" else "FLIM",
                input_type: {"selected_feature_extractors": extractors,
                             "num_components": 1, "input_suffixes": suffixes}},
    }
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(cfg))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    metadata_widgets.clear_folder_scan_caches()
    app = AppTest.from_file(PAGE).run(timeout=30)
    app.text_input(key="fov_metadata_folder_path").set_value(str(folder)).run(timeout=30)
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    return app, folder


@pytest.mark.parametrize("kind", ["intensity", "prefitted"])
def test_workflows_without_shift_calibration_export_real_features(tmp_path, monkeypatch, kind):
    app, folder = no_calibration_app(tmp_path, monkeypatch, kind)
    # Without calibration the first button is the extraction start itself.
    assert not any(b.label == "Start calibration" for b in app.button)
    button(app, "Start extraction").click().run(timeout=30)
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    assert not any(b.label == "Optimize for Shifts" for b in app.button)
    assert not any(w.label == "Fitting Mode" for w in app.selectbox)
    files = list(folder.glob("single_cell_features_*.csv"))
    assert len(files) == 1
    features = pd.read_csv(files[0])
    assert features["cell_id"].tolist() == ["sample_1", "sample_2"]
    if kind == "prefitted":
        assert features["Lifetime fit_NADH: t1"].eq(1200).all()
    run(app)
    assert len(list(folder.glob("single_cell_features_*.csv"))) == 1
    assert not button(app, "Start extraction").disabled  # A rerun stays one click away.
    # This is the view that followed the click, so the saved record stays with the features.
    assert saved_metadata_notes(app)


def test_step_round_trip_restores_source_controls_in_browser(workflow, monkeypatch):
    """The browser remounts hidden widgets from their protos, not from Session State.

    A remounted widget shows ``proto.value`` only when ``set_value`` is True;
    otherwise it shows the constructor default and sends that default on the
    next rerun, which invalidated the prepared session in a real browser.
    """
    from streamlit.elements.lib import policies
    app, folder, _, _ = workflow
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    app.number_input(key="2D_decay_duration").set_value(10.0).run(timeout=30)
    button(app, "Start calibration").click().run(timeout=30)
    app.radio[0].set_value("**Categorical** (e.g. treatment, day)").run(timeout=30)
    app.radio[0].set_value("**Numerical** (e.g. lifetime, morphology)").run(timeout=30)
    assert not app.exception
    assert not app.warning, [w.value for w in app.warning]
    expected = {
        "fov_metadata_folder_path": str(folder),
        "NADH_Decay (2D)_IRF_suffix": "irf.csv",
        "NADH_Decay (2D)_Decay_suffix": "_decay.csv",
    }
    for key, value in expected.items():
        proto = app.text_input(key=key).proto
        assert proto.set_value, key
        assert proto.value == value, key
    proto = app.number_input(key="2D_decay_duration").proto
    assert proto.set_value and proto.value == 10.0
    proto = app.checkbox(key="has_channel_ch1").proto
    assert proto.set_value and proto.value is True
    assert app.session_state["prepared_extraction"] is not None


def saved_metadata_notes(app):
    return [s.value for s in app.success if s.value.startswith("Metadata is saved automatically")]


def test_saved_metadata_note_retires_once_calibration_moves_on(workflow):
    app, _, _, _ = workflow
    button(app, "Start calibration").click().run(timeout=30)
    run(app)
    assert saved_metadata_notes(app)
    button(app, "Optimize for Shifts").click().run(timeout=30)
    run(app)
    assert not saved_metadata_notes(app)
    button(app, CONFIRM).click().run(timeout=30)
    run(app)
    assert not saved_metadata_notes(app)
    button(app, "Start extraction").click().run(timeout=30)
    run(app)
    assert not saved_metadata_notes(app)
    assert any(s.value.startswith("Single cell features exported") for s in app.success)


def test_calibration_replaces_metadata_settings_and_reports_the_saved_record(workflow):
    app, folder, _, _ = workflow
    assert not [e for e in app.expander if e.label == "Metadata settings"]
    assert not [s for s in app.success if s.value.startswith("Metadata is saved automatically")]
    assert app.dataframe  # The metadata view previews the FOV table.
    button(app, "Start calibration").click().run(timeout=30)
    run(app)  # AppTest keeps the pre-rerun elements; mirror the browser first.
    prepared = app.session_state["prepared_extraction"]
    # The calibration view replaces the metadata view in both columns.
    assert not app.dataframe
    assert [s.value for s in app.success] == [f"Metadata is saved automatically to {prepared.metadata_path}"]
    settings = next(e for e in app.expander if e.label == "Metadata settings")
    assert settings.text_input(key="fov_metadata_folder_path").value == str(folder)
    assert settings.checkbox(key="has_channel_ch1").value is True
    assert settings.number_input(key="2D_decay_duration").value == 12.5
    assert button(app, "Optimize for Shifts")
    notes = [s.value for s in app.success if s.value.startswith("Metadata is saved automatically")]
    assert notes == [f"Metadata is saved automatically to {prepared.metadata_path}"]
    assert not any("Metadata is saved" in c.value for c in app.caption)
    # Editing a metadata setting discards the session and brings the settings back out.
    settings.number_input(key="2D_decay_duration").set_value(10.0).run(timeout=30)
    assert not app.exception
    run(app)
    assert app.session_state["prepared_extraction"] is None
    assert not [e for e in app.expander if e.label == "Metadata settings"]
    assert app.number_input(key="2D_decay_duration").value == 10.0
    assert button(app, "Start calibration")
    assert not [s for s in app.success if s.value.startswith("Metadata is saved automatically")]
    assert app.dataframe  # The metadata view is back.


def test_start_extraction_waits_for_a_failed_metadata_save(tmp_path, monkeypatch):
    from src import extraction_session
    app, folder = no_calibration_app(tmp_path, monkeypatch, "intensity")

    def fail(*args):
        raise PermissionError("locked file")

    with monkeypatch.context() as patch:
        patch.setattr(extraction_session.os, "replace", fail)
        button(app, "Start extraction").click().run(timeout=30)
        run(app)
        prepared = app.session_state["prepared_extraction"]
        assert prepared is not None and not prepared.metadata_path.exists()
        assert any("locked file" in e.value for e in app.error)
        assert not list(folder.glob("single_cell_features_*.csv"))
        assert button(app, "Start extraction").disabled
    button(app, "Retry saving metadata").click().run(timeout=30)
    run(app)
    assert prepared.metadata_path.exists()
    assert not list(folder.glob("single_cell_features_*.csv"))
    assert not button(app, "Start extraction").disabled
    button(app, "Start extraction").click().run(timeout=30)
    assert not app.error, [e.value for e in app.error]
    assert len(list(folder.glob("single_cell_features_*.csv"))) == 1
