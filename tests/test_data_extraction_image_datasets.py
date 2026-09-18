"""Opt-in extraction workflows for the external measured image datasets.

Set FLIM_REAL_DATA_ROOT to the flim_playground_example_data folder and run this
module with pytest. All input files are copied before the app opens them; only
the configuration path is redirected. No extraction or fitting code is mocked.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import toml
from sdtfile import SdtFile
from streamlit.testing.v1 import AppTest
from test_data_extraction_real_dataset import (
    _CATEGORICAL_STEP,
    _NUMERIC_STEP,
    _button,
    _check_calibrated_phasors,
    _refresh_after_rerun,
    _step,
)

from src import config

_ROOT = Path(__file__).resolve().parents[1]
_CHANNEL = "NADH"


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _prefitted_reference(folder, fov, mask, labels):
    """Average the original SPCImage pixel maps directly over each mask label."""
    maps = {}
    for feature, suffix in {"t1": "t1", "t2": "t2", "a1": "a1[%]"}.items():
        values = np.loadtxt(folder / "SPCImage_prefitted" / f"{fov}_Ch2_summed_{suffix}.asc", encoding="utf-8-sig")
        assert values.shape == mask.shape
        maps[feature] = np.where(values == 0, np.nan, values)
    fraction = maps["a1"] / 100
    maps["tm"] = fraction * maps["t1"] + (1 - fraction) * maps["t2"]
    maps["tm_iw"] = (fraction * maps["t1"]**2 + (1 - fraction) * maps["t2"]**2) / maps["tm"]
    maps = {f"Lifetime fit_{_CHANNEL}: {key}": value for key, value in maps.items()}
    maps[f"{_CHANNEL}_a2"] = 100 - maps[f"Lifetime fit_{_CHANNEL}: a1"]

    def mean_valid(values):
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

    return {
        f"{fov}_{label}": {
            feature: mean_valid(values[mask == label])
            for feature, values in maps.items()
        }
        for label in labels
    }


@pytest.fixture(scope="module", params=["prefitted", "3d", "4d"])
def image_dataset(request):
    configured_root = os.environ.get("FLIM_REAL_DATA_ROOT")
    if not configured_root:
        pytest.skip("Set FLIM_REAL_DATA_ROOT to the external example-data folder.")
    root = Path(configured_root).expanduser().resolve() / "Data_Extraction"
    kind = request.param
    is_4d = kind == "4d"
    folder = root / ("4d_decay_1channel" if is_4d else "3d_decay_1channel")
    assert folder.is_dir(), folder
    mask_suffix = "_photons_cell_masks.tif" if is_4d else "_Ch2_summed_cellpose.tiff"
    irf_path = folder / ("220913_nadh.csv" if is_4d else "nadh_irf.txt")
    decay_paths = sorted(folder.glob("*.sdt"))
    assert len(decay_paths) == (1 if is_4d else 25)
    result = dict(
        kind=kind, folder=folder, mask_suffix=mask_suffix, irf_path=irf_path,
        input_type="Decay (3/4D) pixel-prefitted" if kind == "prefitted" else "Decay (3/4D)",
        channel=2 if is_4d else -1,
        irf=np.loadtxt(irf_path, delimiter="," if is_4d else None),
        raw_curves={}, cell_ids={}, fov_counts={}, prefitted_reference={}, masks=[],
        decays=decay_paths, component_paths=[], source_sha256={},
    )
    for path in decay_paths:
        fov = path.stem
        mask_path = next(folder.rglob(f"{fov}{mask_suffix}"))
        mask = tifffile.imread(mask_path)
        labels = np.unique(mask)
        labels = labels[labels != 0]
        result["masks"].append(mask_path)
        result["fov_counts"][fov] = len(labels)
        result["cell_ids"][fov] = [f"{fov}_{label}" for label in labels]
        # Decode the original file independently of the application's reader.
        sdt = SdtFile(path)
        assert len(sdt.data) == 1
        decay = sdt.data[0][2] if is_4d else sdt.data[0]
        assert decay.shape == (*mask.shape, 256)
        result["raw_curves"][fov] = np.stack([decay[mask == label].sum(axis=0) for label in labels])
        result["duration"] = float(sdt.measure_info[0].tac_r / sdt.measure_info[0].tac_g * 1e9)
        if kind == "prefitted":
            result["prefitted_reference"].update(_prefitted_reference(folder, fov, mask, labels))
            result["component_paths"].extend([
                folder / "SPCImage_prefitted" / f"{fov}_Ch2_summed_{suffix}.asc"
                for suffix in ("t1", "t2", "a1[%]")
            ])
    assert sum(result["fov_counts"].values()) == (176 if is_4d else 584)
    assert result["irf"].shape == (256,)
    for path in [*result["decays"], *result["masks"], *result["component_paths"], irf_path]:
        result["source_sha256"][str(path)] = _digest(path)
    return result


@pytest.mark.parametrize(("mode", "extractors"), [
    ("fit_only", ["Lifetime fit"]),
    ("fit_free_only", ["Lifetime fit free"]),
    ("fit_and_fit_free", ["Lifetime fit", "Lifetime fit free"]),
])
def test_real_image_extraction_through_configuration_and_csv_export(
    image_dataset, tmp_path, monkeypatch, mode, extractors,
):
    data = image_dataset
    started = time.monotonic()
    summary = {"dataset": data["kind"], "mode": mode, "stage": "copy_inputs"}
    summary["zero_photon_cells"] = [
        cell_id for fov, curves in data["raw_curves"].items()
        for cell_id, total in zip(data["cell_ids"][fov], curves.sum(axis=1)) if total == 0
    ]

    def record(stage):
        summary.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 2))
        (tmp_path / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def run(app, stage, check_errors=True):
        record(stage)
        app.run(timeout=600)
        assert not app.exception, [e.value for e in app.exception]
        if check_errors:
            assert not app.error, [e.value for e in app.error]
        return app

    record("copy_inputs")
    input_type = data["input_type"]
    has_fit = "Lifetime fit" in extractors
    has_fit_free = "Lifetime fit free" in extractors
    prefitted = data["kind"] == "prefitted"
    needs_raw = not (prefitted and mode == "fit_only")
    fits_raw = has_fit and not prefitted
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sources = list(data["masks"])
    if needs_raw:
        sources += [*data["decays"], data["irf_path"]]
    if has_fit and prefitted:
        sources += data["component_paths"]
    for source in sources:
        target = data_dir / source.relative_to(data["folder"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if not needs_raw:
        assert not list(data_dir.rglob("*.sdt"))
        assert not (data_dir / data["irf_path"].name).exists()

    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(config, "_CONFIG_PATH", config_path)
    app = run(AppTest.from_file(str(_ROOT / "main.py")), "open_configuration", False)
    app.selectbox(key="flim_decay_input_type_default").set_value(input_type)
    run(app, "choose_input_format", False)
    app.text_input(key="channel_name_ch1_default").set_value(_CHANNEL)
    app.multiselect(key=f"{input_type}_ch1_feature_extractors_default").set_value(extractors)
    app.multiselect(key="categorical_cols_default").set_value(["acquisition"])
    run(app, "choose_extractors", False)
    assert any(w.key == f"laser_rate_{input_type}_default_mhz" for w in app.number_input) == has_fit_free
    assert any(w.key == f"fit_free_calibration_{input_type}_default" for w in app.radio) == has_fit_free
    if has_fit_free:
        app.number_input(key=f"laser_rate_{input_type}_default_mhz").set_value(80.0)
        app.radio(key=f"fit_free_calibration_{input_type}_default").set_value("IRF")
    if has_fit:
        app.number_input(key=f"num_components_ch1_{input_type}_default").set_value(2)
    suffixes = {"Mask": data["mask_suffix"]}
    if needs_raw:
        suffixes.update(Decay=".sdt", IRF=data["irf_path"].name)
    if has_fit and prefitted:
        suffixes["SPCImage t1"] = "_Ch2_summed_t1.asc"
    for file_type, suffix in suffixes.items():
        app.text_input(key=f"ch1_{input_type}_{file_type}_default").set_value(suffix)
    if not needs_raw:
        assert not any(w.key == f"ch1_{input_type}_{kind}_default" for w in app.text_input for kind in ("Decay", "IRF"))
    _button(app, "Update Configuration").click()
    run(app, "save_configuration")
    saved = toml.load(config_path)["profiles"]["default"]
    assert saved["ch1"][input_type]["selected_feature_extractors"] == extractors
    assert ("laser_rate" in saved[input_type]) == has_fit_free
    assert ("fit_free_calibration" in saved[input_type]) == has_fit_free

    app.switch_page("pages/data_extraction.py")
    run(app, "open_extraction")
    app.text_input(key="fov_metadata_folder_path").set_value(str(data_dir))
    run(app, "scan_all_real_fovs")
    if data["kind"] == "4d":
        # The third acquisition channel is NADH; channel one is empty and two is dim.
        app.selectbox(key=f"{_CHANNEL}_channel_selectbox").set_value(3)
        run(app, "assign_nadh_channel_3")
    app.button(key="prepare_extraction_button").click()
    run(app, "export_metadata")
    metadata_path = Path(app.session_state["prepared_extraction"].metadata_path)
    metadata = pd.read_csv(metadata_path)
    assert set(metadata["image_name"]) == set(data["fov_counts"])
    assert ("laser_rate" in metadata) == has_fit_free
    assert ("fit_free_calibration_method" in metadata) == has_fit_free
    if has_fit_free:
        assert metadata["laser_rate"].eq(0.08).all()
        assert metadata["fit_free_calibration_method"].eq("IRF").all()
    if needs_raw:
        assert metadata["time_bins"].eq(256).all()
        np.testing.assert_allclose(metadata["duration"], data["duration"], rtol=1e-12)
        assert metadata[f"{_CHANNEL}_channel"].eq(data["channel"]).all()
    else:
        assert not {f"{_CHANNEL}_Decay", f"{_CHANNEL}_IRF", "time_bins", "duration"}.intersection(metadata)

    _step(app, _NUMERIC_STEP)
    run(app, "configure_numeric_extraction")
    if needs_raw:
        if fits_raw:
            app.selectbox(key="fitting_metric").set_value("WLS")
        _button(app, "Optimize for Shifts").click()
        run(app, "optimize_irf_shifts")
        _button(app, "Confirm Time Gates (if applicable) and Shift for each channel").click()
        run(app, "confirm_shifts")
        _refresh_after_rerun(app)
        assert not app.error, [e.value for e in app.error]
        if fits_raw:
            app.selectbox(key="fitting_mode_update").set_value("Local")
            run(app, "save_calibrated_metadata")
        metadata = pd.read_csv(metadata_path)
        assert np.isfinite(metadata[f"{_CHANNEL}_shift"]).all()
        _button(app, "Start extraction").click()
        run(app, "extract_all_cells")
    else:
        # Without calibration, the Start extraction click already ran the extraction.
        assert not any(b.label == "Optimize for Shifts" for b in app.button)
        assert not app.selectbox
    summary["extraction_warnings"] = [w.value for w in app.warning]
    paths = list(data_dir.glob("single_cell_features_*.csv"))
    assert len(paths) == 1
    features = pd.read_csv(paths[0])
    summary["rows"] = len(features)
    summary["nan_counts"] = {c: int(n) for c, n in features.isna().sum().items() if n}
    record("validate_numeric_output")
    assert features["cell_id"].is_unique
    assert set(features["cell_id"]) == {cell_id for ids in data["cell_ids"].values() for cell_id in ids}
    assert features["image_name"].value_counts().to_dict() == data["fov_counts"]
    assert any(c.startswith(f"Lifetime fit_{_CHANNEL}:") for c in features) == has_fit
    assert any(c.startswith(f"Lifetime fit free_{_CHANNEL}:") for c in features) == has_fit_free
    numeric = features.select_dtypes(include="number")
    independently_checked = set()
    finite_rows = pd.Series(True, index=features.index)
    if has_fit:
        if prefitted:
            expected = pd.DataFrame.from_dict(data["prefitted_reference"], orient="index")
            actual = features.set_index("cell_id").loc[expected.index, expected.columns]
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-8, equal_nan=True)
            independently_checked.update(expected.columns)
            summary["cells_without_prefitted_pixels"] = expected.index[expected.isna().all(axis=1)].tolist()
        else:
            # The masks include three regions with no recorded photons. Their
            # fits must remain missing; require finite results for every region
            # with signal, including the separate one-photon region.
            finite_rows = ~features["cell_id"].isin(summary["zero_photon_cells"])
            fit_columns = [c for c in numeric if not c.startswith(f"Lifetime fit free_{_CHANNEL}:")]
            assert numeric.loc[~finite_rows, fit_columns].isna().all().all()
            t1 = features.loc[finite_rows, f"Lifetime fit_{_CHANNEL}: t1"]
            t2 = features.loc[finite_rows, f"Lifetime fit_{_CHANNEL}: t2"]
            assert (t1 > 0).all() and (t2 >= t1).all()
            assert (t2 <= data["duration"] * 1000).all()
            assert metadata["fitting_algo"].eq("WLS").all()
            assert metadata["fitting_mode"].eq("Local").all()
    if has_fit_free:
        _check_calibrated_phasors(features, metadata, data["raw_curves"], data["irf"], data["cell_ids"])
        independently_checked.update(c for c in numeric if c.startswith(f"Lifetime fit free_{_CHANNEL}:"))
    finite_columns = [c for c in numeric if c not in independently_checked]
    assert np.isfinite(numeric.loc[finite_rows, finite_columns].to_numpy()).all()

    _step(app, _CATEGORICAL_STEP)
    run(app, "open_categorical_extraction")
    next(w for w in app.text_input if w.label == "Field of View Name Delimiter").set_value("")
    next(w for w in app.text_input if w.label == "Copy the folder path here").set_value(str(data_dir))
    run(app, "load_feature_csv")
    next(w for w in app.multiselect if w.label.startswith("Choose Categorical features")).set_value(["acquisition"])
    run(app, "choose_acquisition_label")
    slot = app.multiselect(key="slot_acquisition")
    slot.set_value([slot.options[0]])
    _button(app, "Confirm category mapping & export the combined dataset").click()
    run(app, "export_labeled_csv")
    combined_paths = list(data_dir.glob("*_combined.csv"))
    assert len(combined_paths) == 1
    combined = pd.read_csv(combined_paths[0])
    pd.testing.assert_frame_equal(combined[features.columns], features)
    assert combined["acquisition"].equals(combined["image_name"])
    for source in sources:
        assert _digest(source) == data["source_sha256"][str(source)]
    summary.update(
        fov_counts=data["fov_counts"], feature_columns=list(features.columns),
        source_sha256={str(p): data["source_sha256"][str(p)] for p in sources},
        shift_bins=metadata.set_index("image_name")[f"{_CHANNEL}_shift"].to_dict() if needs_raw else {},
        metadata_csv=str(metadata_path), features_csv=str(paths[0]),
        categorized_csv=str(combined_paths[0]), config_toml=str(config_path),
    )
    record("passed")
