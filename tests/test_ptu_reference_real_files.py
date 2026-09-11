"""Opt-in, read-only smoke checks for the supplied Atto reference acquisitions.

Set FLIM_PTU_REFERENCE_ROOT to the folder holding both files to run these tests.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ptufile import PtuFile

from src.decay_io import read_decay_metadata, read_ptu_reference
from src.file_io import get_irf, get_lifetime_standard


@pytest.mark.parametrize("name, frames", [
    ("Atto488.ptu", 4),
    ("Quenched Atto 488 500 nM PI 5 uM high repetition.ptu", 160),
])
def test_supplied_reference_read_only(name, frames):
    root = os.environ.get("FLIM_PTU_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set FLIM_PTU_REFERENCE_ROOT to the supplied Atto folder.")
    path = Path(root) / name
    before = path.stat()
    err, ref = read_ptu_reference(path)
    assert err == ""
    assert ref.time_bins == 264
    assert ref.n_frames == frames
    assert ref.laser_rate == pytest.approx(0.03901)
    assert ref.duration / ref.time_bins == pytest.approx(0.09696969697)
    err, duration = read_decay_metadata(str(path))
    assert err == "" and ref.duration == duration

    # Decode individual scan lines then sum in NumPy as an independent check
    # of decoder-side Y integration, without allocating a full image stack.
    with PtuFile(path) as ptu:
        records = ptu.read_records(memmap=True, cache=True)
        lines = ptu.decode_image(
            (slice(None), slice(None), slice(None, None, -1), slice(None), slice(None)),
            records=records, frame=-1, dtime=ref.time_bins, dtype=np.uint64,
        )
    np.testing.assert_array_equal(ref.curve, lines.sum(axis=(0, 1, 2, 3), dtype=np.uint64))
    assert int(ref.curve.sum()) > 0

    row = pd.Series({
        "ch1_IRF": str(path), "ch1_Fluorescence Lifetime Standard": str(path),
        "time_bins": ref.time_bins, "duration": ref.duration,
        "laser_rate": ref.laser_rate,
    })
    for sample_metadata in (row, pd.DataFrame([row])):
        err, irf = get_irf(sample_metadata, "ch1", ref.time_bins)
        assert err == ""
        assert isinstance(irf, np.ndarray)
        assert irf.shape == (ref.time_bins,)
        assert irf.dtype == np.uint64
        assert np.isfinite(irf).all()
        assert (irf >= 0).all()
        np.testing.assert_array_equal(irf, ref.curve)
    for loader in (get_irf, get_lifetime_standard):
        assert loader(row, "ch1", ref.time_bins)[0] == ""
        tissue = row.copy()
        tissue["time_bins"] = 128
        tissue["duration"] = 128 * 0.09696969697
        tissue["laser_rate"] = 0.08
        err, result = loader(tissue, "ch1", 128)
        assert result is None
        assert "time bins" in err and "264" in err and "128" in err
        assert "frequency" in err and "39.01" in err and "80" in err
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    print(f"{name}: {frames} frames, {ref.time_bins} bins, {int(ref.curve.sum())} decoded photons")
