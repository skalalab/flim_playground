"""FLIM 3/4D fit-free channels retain channel_no with lifetime-standard calibration,
even when IRF-shift assignment is skipped.
"""
import pandas as pd
from src import metadata


def test_channel_no_extracted_for_fit_free_standard_3_4d(tmp_path):
    paths = {}
    for kind in ("Decay", "Mask", "Fluorescence Lifetime Standard"):
        path = tmp_path / f"{kind}.tif"
        path.touch()
        paths[f"ch1_{kind}"] = [str(path)]
    fovs = pd.DataFrame({
        **paths, "image_name": ["fov1"], "ch1_channel": [0],
        "time_bins": [4], "duration": [12.5],
        "ch1_fluorescence_lifetime_standard_time_axis": [2],
    })
    err, md = metadata.prepare_extraction(
        fovs, {"ch1": {
            "input_type": "Decay (3/4D)", "imaging_modality": "FLIM",
            "selected_feature_extractors": ["Lifetime fit free"],
        }},
        fov_name_col="image_name", unique_cell_id_col="cell_id", laser_rate=0.08,
        fit_free_calibration_method="Fluorescence Lifetime Standard",
        fluorescence_lifetime_standard_lifetime=4.0,
    )
    assert err == ""
    assert "channel_no" in md["ch1"]
    assert md["ch1"]["channel_no"] == 0
