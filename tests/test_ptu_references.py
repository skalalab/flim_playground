"""Real, small TTTR files exercise the reference decoder and timing contract."""

import os
import struct

import numpy as np
import pandas as pd
import pytest
from ptufile import PtuFile, imwrite
import tifffile

from src import decay_io, file_io


def write_reference(path, data=None, *, frequency=40_000_000, bins=32):
    if data is None:
        data = np.zeros((3, 2, 3, 1, bins), dtype=np.uint16)
        data[0, 0, 0, 0, 3] = 7
        data[1, 1, 2, 0, 4] = 11
        data[2, 0, 1, 0, 5] = 13
    imwrite(path, data, global_resolution=1 / frequency,
            tcspc_resolution=1 / frequency / bins, pixel_time=1e-5)
    return data


def reference_metadata(path, **overrides):
    row = {"ch1_IRF": str(path), "ch1_Fluorescence Lifetime Standard": str(path),
           "time_bins": 32, "duration": 25.0, "laser_rate": 0.04}
    row.update(overrides)
    return pd.Series(row)


def test_reference_integrates_pixels_and_frames_with_period_sized_tail(tmp_path, monkeypatch):
    path = tmp_path / "standard.ptu"
    data = write_reference(path)
    calls = []
    decode = PtuFile.decode_image

    def checked_decode(self, *args, **kwargs):
        assert isinstance(kwargs["records"], np.memmap)
        assert kwargs["records"].mode == "r"
        assert np.dtype(kwargs["dtype"]) == np.uint64
        result = decode(self, *args, **kwargs)
        calls.append(result.shape)
        return result

    monkeypatch.setattr(PtuFile, "decode_image", checked_decode)
    err, ref = decay_io.read_ptu_reference(path)
    assert err == ""
    np.testing.assert_array_equal(ref.curve, data.sum(axis=(0, 1, 2, 3)))
    assert ref.curve.dtype == np.uint64
    assert ref.curve.shape == (32,)
    assert ref.curve.sum() == 31
    assert ref.n_frames == 3
    assert ref.time_bins == 32
    assert ref.duration == pytest.approx(25.0)
    assert ref.laser_rate == pytest.approx(0.04)
    assert calls and np.prod(calls[0]) == 32  # only a compact histogram allocated


@pytest.mark.parametrize("kind, message", [
    ("empty", "empty"), ("multi_channel", "single-channel"),
    ("point", "imaging"), ("t2", "T3"),
])
def test_reference_rejects_unsupported_real_ptus(tmp_path, kind, message):
    path = tmp_path / f"{kind}.ptu"
    data = np.zeros((2, 2, 3, 2 if kind == "multi_channel" else 1, 32), dtype=np.uint16)
    if kind != "empty":
        data[..., 3] = 1
    write_reference(path, data)
    # Change acquisition tags in a real binary fixture to describe point/T2 data.
    if kind in ("point", "t2"):
        raw = bytearray(path.read_bytes())
        tag = b"Measurement_SubMode" if kind == "point" else b"Measurement_Mode"
        struct.pack_into("<q", raw, raw.index(tag) + 40, 0 if kind == "point" else 2)
        path.write_bytes(raw)
    err, ref = decay_io.read_ptu_reference(path)
    assert ref is None
    assert message in err


@pytest.mark.parametrize("contents", [None, b"", b"not a PTU"])
def test_reference_reports_invalid_file_without_raising(tmp_path, contents):
    path = tmp_path / "invalid.ptu"
    if contents is not None:
        path.write_bytes(contents)
    err, ref = decay_io.read_ptu_reference(path)
    assert err and ref is None


@pytest.mark.parametrize("loader", ["get_irf", "get_lifetime_standard"])
def test_reference_loaders_report_non_path_metadata(loader):
    err, result = getattr(file_io, loader)(reference_metadata(123), "ch1", 32)
    assert err and result is None
    row = reference_metadata("unused")
    row["ch1_IRF"] = row["ch1_Fluorescence Lifetime Standard"] = 123
    err, result = getattr(file_io, loader)(row, "ch1", 32)
    assert err and result is None


@pytest.mark.parametrize("loader", ["get_irf", "get_lifetime_standard"])
def test_reference_loaders_match_equivalent_existing_formats(tmp_path, loader):
    path = tmp_path / "reference.ptu"
    data = write_reference(path)
    curve = data.sum(axis=(0, 1, 2, 3))
    old_path = tmp_path / ("reference.csv" if loader == "get_irf" else "reference.tif")
    if loader == "get_irf":
        np.savetxt(old_path, curve, delimiter=",")
    else:
        tifffile.imwrite(old_path, curve.reshape(1, 1, -1), photometric="minisblack")
    load = getattr(file_io, loader)
    err, actual = load(reference_metadata(path), "ch1", 32)
    assert err == ""
    err, expected = load(reference_metadata(old_path), "ch1", 32)
    assert err == ""
    if loader == "get_lifetime_standard":
        assert actual[1] == expected[1] == 2
        assert actual[0].shape == (1, 1, 32)
        np.testing.assert_array_equal(actual[0], expected[0])
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("loader", ["get_irf", "get_lifetime_standard"])
@pytest.mark.parametrize("overrides, time_bins, terms", [
    ({"time_bins": 16}, 16, ("time bins", "32", "16")),
    ({"duration": 12.5}, 32, ("bin width", "0.78125", "0.390625")),
    ({"laser_rate": 0.08}, 32, ("frequency", "40", "80")),
    ({"duration": np.nan}, 32, ("duration",)),
])
def test_reference_timing_mismatch_reports_both_values(tmp_path, loader, overrides, time_bins, terms):
    path = tmp_path / "reference.ptu"
    write_reference(path)
    err, result = getattr(file_io, loader)(reference_metadata(path, **overrides), "ch1", time_bins)
    assert result is None
    for term in terms:
        assert term in err


@pytest.mark.parametrize("loader", ["get_irf", "get_lifetime_standard"])
def test_timing_checks_all_metadata_rows(tmp_path, loader):
    path = tmp_path / "reference.ptu"
    write_reference(path)
    rows = pd.DataFrame([reference_metadata(path), reference_metadata(path, laser_rate=0.08)])
    err, result = getattr(file_io, loader)(rows, "ch1", 32)
    assert result is None and "frequency" in err


@pytest.mark.parametrize("laser_rate", [None, np.nan, 0.04 * (1 + 0.5e-5)])
def test_frequency_optional_and_timing_tolerates_roundoff(tmp_path, laser_rate):
    path = tmp_path / "reference.ptu"
    write_reference(path)
    row = reference_metadata(path, duration=25 * (1 + 0.5e-5), laser_rate=laser_rate)
    err, irf = file_io.get_irf(row, "ch1", 32)
    assert err == ""
    assert irf.sum() == 31


def test_reference_cache_uses_resolved_path_size_mtime_and_rescan(tmp_path, monkeypatch):
    from src.widgets.metadata_widgets import clear_folder_scan_caches

    path = tmp_path / "reference.ptu"
    write_reference(path)
    alias = tmp_path / "alias.ptu"
    alias.symlink_to(path)
    decode = PtuFile.decode_image
    calls = []

    def counted_decode(self, *args, **kwargs):
        calls.append(self.filename)
        return decode(self, *args, **kwargs)

    monkeypatch.setattr(PtuFile, "decode_image", counted_decode)
    for filename in (path, alias):
        err, ref = decay_io.read_ptu_reference(filename)
        assert err == ""
        ref.curve[:] = 0  # callers cannot modify the cached copy
    assert len(calls) == 1
    err, ref = decay_io.read_ptu_reference(path)
    assert err == "" and ref.curve.sum() == 31
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert decay_io.read_ptu_reference(path)[0] == ""
    assert len(calls) == 2
    stat = path.stat()
    with path.open("ab") as stream:
        stream.write(b"\0\0\0\0")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert decay_io.read_ptu_reference(path)[0] == ""
    assert len(calls) == 3
    clear_folder_scan_caches()
    assert decay_io.read_ptu_reference(path)[0] == ""
    assert len(calls) == 4
