"""Step 1 records which PTU files had their repeated frames summed and warns once,
naming each file a single time even when two channels read the same file."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.widgets.metadata_widgets as mw

SINGLE = "/data/fov_a.ptu"
STACK = "/data/fov_b.ptu"
FRAMES = {SINGLE: 1, STACK: 4}


@pytest.fixture(autouse=True)
def _fresh_scan(monkeypatch):
    """The scan is st.cache_data'd: identical (paths, labels) across tests would replay a stale result.
    Only the file readers are faked; the scan and the warning helper run for real."""
    mw._scan_decay_files.clear()
    monkeypatch.setattr(mw, "get_fov_name_col", lambda: "image_name")
    monkeypatch.setattr(mw, "read_decay_metadata", lambda path: ("", 12.5))
    monkeypatch.setattr(
        mw, "read_decay_with_frames",
        lambda path, channel=-1: ("", (np.ones((2, 4, 4, 8), dtype=np.uint32), FRAMES[path])),
    )
    yield
    mw._scan_decay_files.clear()


def _fov_df(paths, channels=("NADH",)):
    df = pd.DataFrame({"image_name": [os.path.basename(p)[:-4] for p in paths]})
    for ch in channels:
        df[f"{ch}_Decay"] = list(paths)
    return df


def test_scan_records_only_files_whose_frames_were_summed():
    err, scan = mw._scan_decay_files((SINGLE, STACK), ("fov_a", "fov_b"))
    assert err == ""
    assert scan.frames_to_files == {4: [STACK]}
    assert scan.shape_list == [(2, 4, 4, 8), (2, 4, 4, 8)]      # the rest of the scan is intact
    assert scan.laser_rep_time_list == [12.5, 12.5]


def test_no_warning_when_every_file_is_a_single_frame():
    assert mw.decay_frame_warning(_fov_df([SINGLE]), ["NADH"]) == ""


def test_warning_names_the_summed_file_and_its_frame_count():
    msg = mw.decay_frame_warning(_fov_df([SINGLE, STACK]), ["NADH"])
    assert "fov_b.ptu" in msg
    assert "4 frames" in msg          # "4 frames (1 file(s))", not "Frames 4"
    assert "fov_a.ptu" not in msg


def test_two_channels_sharing_a_multi_detector_file_list_it_once():
    df = _fov_df([SINGLE, STACK], channels=("NADH", "FAD"))
    msg = mw.decay_frame_warning(df, ["NADH", "FAD"])
    assert msg.count("fov_b.ptu") == 1


def _two_channel_step_one(df):
    """Drive Step 1's channel check the way the page does, with both FLIM channels
    reading the same multi-detector files; masks only need to be present as columns."""
    df["NADH_Mask"] = df["FAD_Mask"] = "mask.tiff"
    return mw.check_assign_channel_widget(
        df,
        {"ch1": "NADH", "ch2": "FAD"},
        "Decay (3/4D)",
        {"ch1": "FLIM", "ch2": "FLIM"},
        {"ch1": ["Lifetime fit"], "ch2": ["Lifetime fit"]},
    )


def test_step_one_warns_once_about_summed_frames_even_with_two_channels(monkeypatch):
    shown = []
    monkeypatch.setattr(mw.st, "warning", lambda msg, *a, **k: shown.append(msg))
    err, out = _two_channel_step_one(_fov_df([SINGLE, STACK], channels=("NADH", "FAD")))
    assert err == ""
    assert out["time_bins"].iloc[0] == 8
    assert len(shown) == 1
    assert shown[0].count("fov_b.ptu") == 1


def test_step_one_stays_quiet_when_every_file_is_a_single_frame(monkeypatch):
    shown = []
    monkeypatch.setattr(mw.st, "warning", lambda msg, *a, **k: shown.append(msg))
    err, _ = _two_channel_step_one(_fov_df([SINGLE], channels=("NADH", "FAD")))
    assert err == ""
    assert shown == []
