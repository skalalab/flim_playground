"""Run the complete extraction UI on the bundled, measured T-cell decays.

No file readers, calibration routines, fitting routines, or extraction outputs are
mocked. Only the config path is redirected; raw data and exports use private copies.
Run explicitly with pytest and --basetemp to retain the CSVs and result summaries.
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toml
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Widget

from src import config
from src.fit_helper import irf_shift

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "example_data/Data_Extraction/T_cell_activation"
_INPUT = "Decay (2D)"
_CHANNEL = "NADH"
_FOV_COUNTS = {"Tcell_Act": 342, "Tcell_Qui": 351, "Tcell_bckgrnd": 103}
_NUMERIC_STEP = "**Numerical** (e.g. lifetime, morphology)"
_CATEGORICAL_STEP = "**Categorical** (e.g. treatment, day)"


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _step(app, label):
    selector = next(r for r in app.radio if r.label == "Select a step to extract single-object features")
    selector.set_value(label)


def _refresh_after_rerun(app):
    # AppTest retains removed widgets in its tree across st.rerun(), even though
    # Streamlit has correctly cleaned up their state. Send only the live widget
    # values, as the browser would, to refresh the tree without losing selection.
    states = WidgetStates()
    for node in app:
        if isinstance(node, Widget) and node.id in app.session_state:
            states.widgets.append(node._widget_state)
    app._run(states, timeout=600)
    assert not app.exception, [e.value for e in app.exception]


def _check_calibrated_phasors(features, metadata, raw_curves, irf, cell_ids=None):
    """Check phasor exports against direct complex Fourier sums.

    The shared IRF interpolation helper supplies the calibrated instrument curve;
    background correction, Fourier coefficients, and lifetimes are recomputed
    from the original measurements independently of the extraction routines.
    """
    prefix = f"Lifetime fit free_{_CHANNEL}: "
    indexed_features = features.set_index("cell_id")
    for row in metadata.to_dict("records"):
        fov = row["image_name"]
        raw = raw_curves[fov]
        tail_start = int(raw.shape[1] * 0.9)
        corrected = np.maximum(raw - raw[:, tail_start:].mean(axis=1, keepdims=True), 0)
        angular_frequency = 2 * np.pi * row["laser_rate"]
        times = np.arange(raw.shape[1]) * row["duration"] / raw.shape[1]
        basis = np.exp(1j * angular_frequency * times[:, None] * np.array([1, 2]))
        with np.errstate(invalid="ignore"):
            signal = corrected @ basis / corrected.sum(axis=1, keepdims=True)
        shifted_irf = irf_shift(irf, row[f"{_CHANNEL}_shift"])
        reference = shifted_irf @ basis / shifted_irf.sum()
        calibrated = signal / reference
        ids = cell_ids[fov] if cell_ids is not None else [f"{fov}_{i}" for i in range(len(raw))]
        exported = indexed_features.loc[ids]
        for index, harmonic in enumerate(("1st", "2nd")):
            np.testing.assert_allclose(
                exported[prefix + f"G({harmonic})"], calibrated[:, index].real,
                rtol=1e-11, atol=1e-12,
            )
            np.testing.assert_allclose(
                exported[prefix + f"S({harmonic})"], calibrated[:, index].imag,
                rtol=1e-11, atol=1e-12,
            )
        first = calibrated[:, 0]
        np.testing.assert_allclose(
            exported[prefix + "Tau_phase"], first.imag / first.real / angular_frequency,
            rtol=1e-11, atol=1e-12,
        )
        with np.errstate(invalid="ignore"):
            tau_mod = np.sqrt(1 / np.abs(first)**2 - 1) / angular_frequency
        np.testing.assert_allclose(
            exported[prefix + "Tau_mod"], tau_mod, rtol=1e-11, atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize(("mode", "lifetime_extractors"), [
    ("fit_only", ["Lifetime fit"]),
    ("fit_free_only", ["Lifetime fit free"]),
    ("fit_and_fit_free", ["Lifetime fit", "Lifetime fit free"]),
])
def test_real_tcell_extraction_from_configuration_through_export(
    tmp_path, monkeypatch, mode, lifetime_extractors,
):
    started = time.monotonic()
    summary = {"mode": mode, "stage": "setup", "source_sha256": {}}

    def record(stage):
        summary.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 2))
        (tmp_path / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def run(app, stage):
        record(stage)
        app.run(timeout=600)
        assert not app.exception, [e.value for e in app.exception]
        return app

    def no_errors(app):
        assert not app.error, [e.value for e in app.error]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    expected_sums = {}
    raw_curves = {}
    for source in sorted(_DATA.iterdir()):
        if source.suffix not in {".csv", ".txt"}:
            continue
        source_bytes = source.read_bytes()
        summary["source_sha256"][source.name] = hashlib.sha256(source_bytes).hexdigest()
        shutil.copy2(source, data_dir / source.name)
        if source.suffix == ".csv":
            raw = np.loadtxt(source, delimiter=",")
            fov = source.name.removesuffix("_filtered.csv")
            assert raw.shape == (_FOV_COUNTS[fov], 200)
            raw_curves[fov] = raw
            expected_sums.update({f"{fov}_{i}": total for i, total in enumerate(raw.sum(axis=1))})

    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(config, "_CONFIG_PATH", config_path)
    has_fit = "Lifetime fit" in lifetime_extractors
    has_fit_free = "Lifetime fit free" in lifetime_extractors
    extractors = lifetime_extractors + ["Intensity texture"]

    app = run(AppTest.from_file(str(_ROOT / "main.py")), "configure")
    app.selectbox(key="flim_decay_input_type_default").set_value(_INPUT)
    run(app, "choose_2d_input")
    app.text_input(key="channel_name_ch1_default").set_value(_CHANNEL)
    app.multiselect(key=f"{_INPUT}_ch1_feature_extractors_default").set_value(extractors)
    app.number_input(key=f"{_INPUT}_duration_default").set_value(12.5)
    app.number_input(key=f"{_INPUT}_time_bins_default").set_value(200)
    app.multiselect(key="categorical_cols_default").set_value(["condition"])
    run(app, "select_extractors")
    assert bool([w for w in app.number_input if w.key == f"laser_rate_{_INPUT}_default_mhz"]) == has_fit_free
    assert bool([w for w in app.radio if w.key == f"fit_free_calibration_{_INPUT}_default"]) == has_fit_free
    heading = next(h for h in app.subheader if h.value == "Shared FLIM settings")
    assert heading.proto.help == "Applies to all FLIM channels."
    assert not [c for c in app.caption if c.value == "Applies to all FLIM channels."]
    if has_fit:
        app.number_input(key=f"num_components_ch1_{_INPUT}_default").set_value(2)
    if has_fit_free:
        app.number_input(key=f"laser_rate_{_INPUT}_default_mhz").set_value(80.0)
        app.radio(key=f"fit_free_calibration_{_INPUT}_default").set_value("IRF")
    app.text_input(key=f"ch1_{_INPUT}_Decay_default").set_value("_filtered.csv")
    app.text_input(key=f"ch1_{_INPUT}_IRF_default").set_value("IRF.txt")
    _button(app, "Update Configuration").click()
    run(app, "save_configuration")
    no_errors(app)
    saved = toml.load(config_path)["profiles"]["default"]
    assert saved["ch1"][_INPUT]["selected_feature_extractors"] == extractors
    assert ("laser_rate" in saved[_INPUT]) == has_fit_free
    assert ("fit_free_calibration" in saved[_INPUT]) == has_fit_free

    app.switch_page("pages/data_extraction.py")
    run(app, "open_extraction")
    app.text_input(key="fov_metadata_folder_path").set_value(str(data_dir))
    run(app, "scan_real_files")
    no_errors(app)
    app.button(key="prepare_extraction_button").click()
    run(app, "export_metadata")
    no_errors(app)
    metadata_path = Path(app.session_state["prepared_extraction"].metadata_path)
    metadata = pd.read_csv(metadata_path)
    assert set(metadata["image_name"]) == set(_FOV_COUNTS)
    assert metadata["duration"].eq(12.5).all()
    assert metadata["time_bins"].eq(200).all()
    assert ("laser_rate" in metadata) == has_fit_free
    assert ("fit_free_calibration_method" in metadata) == has_fit_free
    if has_fit_free:
        assert metadata["laser_rate"].eq(0.08).all()
        assert metadata["fit_free_calibration_method"].eq("IRF").all()

    _step(app, _NUMERIC_STEP)
    run(app, "configure_numeric_extraction")
    no_errors(app)
    if has_fit:
        app.selectbox(key="fitting_metric").set_value("WLS")
    _button(app, "Optimize for Shifts").click()
    run(app, "optimize_real_irf_shifts")
    no_errors(app)
    _button(app, "Confirm calibration for each channel").click()
    run(app, "confirm_shifts")
    no_errors(app)
    _refresh_after_rerun(app)
    no_errors(app)
    if has_fit:
        app.selectbox(key="fitting_mode_update").set_value("Local")
    run(app, "save_calibrated_metadata")
    no_errors(app)
    metadata = pd.read_csv(metadata_path)
    assert np.isfinite(metadata[f"{_CHANNEL}_shift"]).all()
    if has_fit:
        assert metadata["fitting_algo"].eq("WLS").all()
        assert metadata["fitting_mode"].eq("Local").all()
        assert metadata[f"{_CHANNEL}_num_components"].eq(2).all()

    _button(app, "Start extraction").click()
    run(app, "extract_all_796_curves")
    no_errors(app)
    summary["extraction_warnings"] = [w.value for w in app.warning]
    feature_paths = list(data_dir.glob("single_cell_features_*.csv"))
    assert len(feature_paths) == 1
    features = pd.read_csv(feature_paths[0])
    assert len(features) == sum(_FOV_COUNTS.values())
    assert features["cell_id"].is_unique
    assert set(features["cell_id"]) == set(expected_sums)
    assert features["image_name"].value_counts().to_dict() == _FOV_COUNTS
    np.testing.assert_allclose(
        features[f"Intensity texture_{_CHANNEL}: intensity_sum"],
        features["cell_id"].map(expected_sums), rtol=1e-12, atol=1e-6,
    )
    assert any(c.startswith(f"Lifetime fit_{_CHANNEL}:") for c in features) == has_fit
    assert any(c.startswith(f"Lifetime fit free_{_CHANNEL}:") for c in features) == has_fit_free
    numeric = features.select_dtypes(include="number")
    finite_columns = [c for c in numeric if not c.endswith(": Tau_mod")]
    assert np.isfinite(numeric[finite_columns].to_numpy()).all()
    if has_fit:
        t1 = features[f"Lifetime fit_{_CHANNEL}: t1"]
        t2 = features[f"Lifetime fit_{_CHANNEL}: t2"]
        assert (t1 > 0).all() and (t2 >= t1).all()
        assert (t2 <= 12500).all()  # ns input duration -> ps lifetime output.
    if has_fit_free:
        prefix = f"Lifetime fit free_{_CHANNEL}: "
        modulation = np.hypot(features[prefix + "G(1st)"], features[prefix + "S(1st)"])
        valid_modulation = (modulation > 0) & (modulation < 1)
        assert np.isfinite(features[prefix + "Tau_mod"]).equals(valid_modulation)
        _check_calibrated_phasors(features, metadata, raw_curves, np.loadtxt(_DATA / "IRF.txt"))

    # Complete the final extraction step using the real FOV-name condition slot.
    _step(app, _CATEGORICAL_STEP)
    run(app, "open_categorical_extraction")
    next(w for w in app.text_input if w.label == "Copy the folder path here").set_value(str(data_dir))
    run(app, "load_extracted_features")
    no_errors(app)
    next(w for w in app.multiselect if w.label.startswith("Choose Categorical features")).set_value(["condition"])
    run(app, "choose_condition_category")
    condition_slot = app.multiselect(key="slot_condition")
    condition_slot.set_value([condition_slot.options[1]])
    _button(app, "Confirm category mapping & export the combined dataset").click()
    run(app, "export_categorized_features")
    no_errors(app)
    combined_paths = list(data_dir.glob("*_combined.csv"))
    assert len(combined_paths) == 1
    combined = pd.read_csv(combined_paths[0])
    pd.testing.assert_frame_equal(combined[features.columns], features)
    assert combined["condition"].value_counts().to_dict() == {
        fov.split("_")[1]: count for fov, count in _FOV_COUNTS.items()
    }

    summary.update(
        rows=len(features), fov_counts=features["image_name"].value_counts().to_dict(),
        feature_columns=list(features.columns),
        nan_counts={c: int(n) for c, n in numeric.isna().sum().items() if n},
        shift_bins=metadata.set_index("image_name")[f"{_CHANNEL}_shift"].to_dict(),
        metadata_csv=str(metadata_path), features_csv=str(feature_paths[0]),
        categorized_csv=str(combined_paths[0]), config_toml=str(config_path),
    )
    for name, digest in summary["source_sha256"].items():
        assert hashlib.sha256((_DATA / name).read_bytes()).hexdigest() == digest
    record("passed")
