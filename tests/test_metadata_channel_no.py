"""A FLIM 3/4D-decay channel using 'Lifetime fit free' with a per-channel
fluorescence-lifetime-standard calibration still needs its `channel_no`
extracted. A `continue` that was meant only to skip IRF-shift assignment must
not also skip channel-number parsing — otherwise `fov_extraction` later raises
a KeyError on `metadata_dict[channel]["channel_no"]`.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.metadata as metadata_mod
from src.metadata import get_ch_info


def _patch_config(monkeypatch):
    monkeypatch.setattr(
        metadata_mod,
        "get_available_feature_extractors",
        lambda input_type: [
            "Lifetime fit",
            "Lifetime fit free",
            "Intensity morphology",
            "Intensity texture",
        ],
    )
    monkeypatch.setattr(metadata_mod, "get_fov_name_col", lambda: "image_name")
    monkeypatch.setattr(metadata_mod, "get_unique_cell_id_col", lambda: "cell_id")


def _fit_free_standard_3_4d_metadata():
    return pd.DataFrame(
        {
            "ch1_input_type": ["Decay (3/4D)"],
            "ch1_imaging_modality": ["FLIM"],
            "ch1_Lifetime fit free": [True],
            "ch1_Fluorescence Lifetime Standard": ["standard.tif"],
            "ch1_fluorescence_lifetime_standard_time_axis": ["0,1,2,3"],
            "ch1_channel": [0],
            "laser_rate": [80.0],
            "fit_free_calibration_method": ["Fluorescence Lifetime Standard"],
            "fluorescence_lifetime_standard_lifetime": [4.0],
        }
    )


def test_channel_no_extracted_for_fit_free_standard_3_4d(monkeypatch):
    _patch_config(monkeypatch)
    err, md = get_ch_info(_fit_free_standard_3_4d_metadata())
    assert err == ""
    assert "channel_no" in md["ch1"]
    assert md["ch1"]["channel_no"] == 0
