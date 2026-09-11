"""PTU reading preserves channel, pixel, and time-bin coordinates, sums repeated
frames, and sizes the time axis by the laser period rather than by the highest
bin that happened to receive a photon.

A mocked (T,Y,X,C,H) buffer encodes each voxel's coordinates and verifies conversion
to separate (Y,X,H) channel arrays without requiring a binary fixture.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import decay_io
from src.decay_io import read_decay, read_ptu

Y, X, H = 6, 7, 8          # all < 100 so the encoding below is uniquely decodable


def _buffer(nc, nframes=1):
    """(nframes, Y, X, nc, H) array; every frame's voxel = c*1e6 + y*1e4 + x*100 + h."""
    cc = np.arange(nc).reshape(1, 1, 1, nc, 1)
    yy = np.arange(Y).reshape(1, Y, 1, 1, 1)
    xx = np.arange(X).reshape(1, 1, X, 1, 1)
    hh = np.arange(H).reshape(1, 1, 1, 1, H)
    frame = (cc * 1_000_000 + yy * 10_000 + xx * 100 + hh).astype(np.int64)
    return np.repeat(frame, nframes, axis=0)


def _expected(ch):
    """What channel ch's (Y, X, H) image must contain if nothing is scrambled."""
    yy = np.arange(Y).reshape(Y, 1, 1)
    xx = np.arange(X).reshape(1, X, 1)
    hh = np.arange(H).reshape(1, 1, H)
    return ch * 1_000_000 + yy * 10_000 + xx * 100 + hh


class _FakePtu:
    """Mimics ptufile.PtuFile for a T3 imaging file with canonical dims (T,Y,X,C,H).

    The laser runs at 1 Hz (period 1 s) and the TCSPC resolution is
    1/bins_in_period, so the period holds exactly ``bins_in_period`` bins; keep it a
    power of two so the division is exact. Like ptufile, ``decode_image`` accumulates
    in the requested dtype (uint16 by default) and wraps on overflow, and ``shape[-1]``
    is the highest bin that received a photon, not the period.
    """
    dims = ("T", "Y", "X", "C", "H")

    def __init__(self, nc, nframes=1, bins_in_period=H, buffer=None):
        self._buf = _buffer(nc, nframes) if buffer is None else buffer
        self.shape = self._buf.shape                 # (T, Y, X, nc, H)
        self.number_bins = self.shape[-1]
        self.tags = {"TTResult_SyncRate": 1.0}
        self.tcspc_resolution = 1.0 / bins_in_period
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def decode_image(self, selection=None, *, frame=None, dtime=None, dtype=None, keepdims=True, **_):
        self.calls.append({"frame": frame, "dtime": dtime, "dtype": dtype})
        data = self._buf
        if frame is not None:
            data = data.sum(axis=0, keepdims=True) if frame < 0 else data[frame:frame + 1]
        if dtime is not None and dtime > 0:
            if dtime <= data.shape[-1]:
                data = data[..., :dtime]
            else:
                pad = np.zeros(data.shape[:-1] + (dtime - data.shape[-1],), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=-1)
        return data.astype(np.uint16 if dtype is None else dtype)   # wraps like ptufile


def test_read_ptu_two_channels_not_scrambled(monkeypatch):
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(2))
    for ch in (0, 1):
        err, got = read_ptu("dummy.ptu", channel=ch)
        assert err == ""
        assert got.shape == (Y, X, H)
        np.testing.assert_array_equal(
            got, _expected(ch),
            err_msg=f"read_ptu channel {ch} is scrambled (channel/pixel/time mismatch)")

    # Verify the coordinate fixture distinguishes an axis move from a blind reshape.
    buf0 = _FakePtu(2)._buf[0]            # (Y, X, C, H)
    old = buf0.reshape(2, Y, X, H)[0]     # the buggy operation
    assert not np.array_equal(old, _expected(0)), "test dims too small to expose the bug"


def test_read_ptu_single_channel_unaffected(monkeypatch):
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1))
    err, got = read_ptu("dummy.ptu", channel=-1)   # single-channel sentinel
    assert err == ""
    assert got.shape == (Y, X, H)
    np.testing.assert_array_equal(got, _expected(0))


def test_read_ptu_sums_repeated_frames(monkeypatch):
    # Three scans of the same field: each voxel's coordinate code appears three times over.
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(2, nframes=3))
    for ch in (0, 1):
        err, got = read_ptu("dummy.ptu", channel=ch)
        assert err == ""
        assert got.shape == (Y, X, H)
        np.testing.assert_array_equal(got, 3 * _expected(ch))


def test_read_ptu_summing_frames_does_not_wrap_at_16_bits(monkeypatch):
    # 3 frames x 30,000 photons per voxel = 90,000, past uint16's 65,535 ceiling.
    buf = np.full((3, Y, X, 1, H), 30_000, dtype=np.int64)
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1, buffer=buf))
    err, got = read_ptu("dummy.ptu")
    assert err == ""
    assert got.min() == 90_000 and got.max() == 90_000


def test_read_ptu_keeps_only_the_bins_inside_the_laser_period(monkeypatch):
    # 8 bins received photons but the period holds 4: bins 4..7 lie past the next sync pulse.
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1, bins_in_period=4))
    err, got = read_ptu("dummy.ptu")
    assert err == ""
    assert got.shape == (Y, X, 4)
    np.testing.assert_array_equal(got, _expected(0)[..., :4])


def test_read_ptu_pads_the_time_axis_when_the_period_outlasts_recorded_bins(monkeypatch):
    # Only 8 bins received photons but the period holds 16: the tail is real, empty time.
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1, bins_in_period=16))
    err, got = read_ptu("dummy.ptu")
    assert err == ""
    assert got.shape == (Y, X, 16)
    np.testing.assert_array_equal(got[..., :H], _expected(0))
    assert not got[..., H:].any()


def test_read_decay_with_frames_reports_how_many_frames_were_summed(monkeypatch, tmp_path):
    from src.decay_io import read_decay_with_frames
    path = tmp_path / "stack.ptu"
    path.write_bytes(b"")                 # only the path check touches the disk; PtuFile is faked
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1, nframes=4))
    err, result = read_decay_with_frames(str(path))
    assert err == ""
    data, n_frames = result
    assert n_frames == 4
    np.testing.assert_array_equal(data, 4 * _expected(0))
    err, plain = read_decay(str(path))   # the frame-less reader sees the same summed image
    assert err == ""
    np.testing.assert_array_equal(plain, data)


def test_read_ptu_reports_a_non_image_file_instead_of_raising(monkeypatch):
    fake = _FakePtu(1)
    fake.dims, fake.shape = (), ()        # what ptufile exposes for T2 / point records
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: fake)
    err, got = read_ptu("dummy.ptu")
    assert err != ""
    assert got is None


def test_read_ptu_reports_a_missing_tcspc_resolution_instead_of_raising(monkeypatch):
    fake = _FakePtu(1)
    fake.tcspc_resolution = 0.0           # ptufile returns 0.0 when MeasDesc_Resolution is absent
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: fake)
    err, got = read_ptu("dummy.ptu")
    assert got is None
    assert "resolution" in err.lower()


# --- Leica FALCON exports: a rounded 80 MHz sync tag beside 96.97 ps TDC bins ---------
_FALCON_SYNC_RATE = 80_000_000
_FALCON_RESOLUTION = 9.696969697e-11      # seconds; 1 / sync rate holds 128.9 of these bins


def _falcon_ptu(**overrides):
    fake = _FakePtu(1)
    fake.tags = {"TTResult_SyncRate": _FALCON_SYNC_RATE}
    fake.tcspc_resolution = _FALCON_RESOLUTION
    for name, value in overrides.items():
        setattr(fake, name, value)
    return fake


def test_ptu_duration_over_time_bins_is_the_tcspc_resolution(monkeypatch, tmp_path):
    """Step 1 stores read_decay_metadata's value as ``duration`` and the fit divides it
    by the bin count, so the two must share the resolution: 128 bins of 96.97 ps span
    12.412 ns. Reporting the nominal 12.5 ns period instead stretches every bin to
    97.66 ps, and every fitted lifetime by the same 0.7 %."""
    path = tmp_path / "falcon.ptu"
    path.write_bytes(b"")
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _falcon_ptu())
    err, duration_ns = decay_io.read_decay_metadata(str(path))
    assert err == ""
    err, decay = read_ptu(str(path))
    assert err == ""
    time_bins = decay.shape[-1]
    assert time_bins == 128
    np.testing.assert_allclose(duration_ns / time_bins, _FALCON_RESOLUTION * 1e9, rtol=1e-9)


def test_read_decay_metadata_reports_a_missing_tcspc_resolution_instead_of_raising(monkeypatch, tmp_path):
    path = tmp_path / "falcon.ptu"
    path.write_bytes(b"")
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _falcon_ptu(tcspc_resolution=0.0))
    err, duration_ns = decay_io.read_decay_metadata(str(path))
    assert duration_ns is None
    assert "resolution" in err.lower()
