"""PTU reading preserves channel, pixel, and time-bin coordinates.
A mocked (T,Y,X,C,H) buffer encodes each voxel's coordinates and verifies conversion
to separate (Y,X,H) channel arrays without requiring a binary fixture.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.decay_io as decay_io
from src.decay_io import read_ptu

Y, X, H = 6, 7, 8          # all < 100 so the encoding below is uniquely decodable


def _buffer(nc):
    """(1, Y, X, nc, H) array where voxel = c*1e6 + y*1e4 + x*100 + h."""
    cc = np.arange(nc).reshape(1, 1, 1, nc, 1)
    yy = np.arange(Y).reshape(1, Y, 1, 1, 1)
    xx = np.arange(X).reshape(1, 1, X, 1, 1)
    hh = np.arange(H).reshape(1, 1, 1, 1, H)
    return (cc * 1_000_000 + yy * 10_000 + xx * 100 + hh).astype(np.int64)


def _expected(ch):
    """What channel ch's (Y, X, H) image must contain if nothing is scrambled."""
    yy = np.arange(Y).reshape(Y, 1, 1)
    xx = np.arange(X).reshape(1, X, 1)
    hh = np.arange(H).reshape(1, 1, H)
    return ch * 1_000_000 + yy * 10_000 + xx * 100 + hh


class _FakePtu:
    """Mimics ptufile.PtuFile for an imaging file with canonical dims (T,Y,X,C,H)."""
    dims = ("T", "Y", "X", "C", "H")

    def __init__(self, nc):
        self._buf = _buffer(nc)
        self.shape = self._buf.shape  # (1, Y, X, nc, H)

    def __getitem__(self, key):
        return self._buf[key]         # ptu[0] -> (Y, X, nc, H)


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
    buf0 = _FakePtu(2)[0]                 # (Y, X, C, H)
    old = buf0.reshape(2, Y, X, H)[0]     # the buggy operation
    assert not np.array_equal(old, _expected(0)), "test dims too small to expose the bug"


def test_read_ptu_single_channel_unaffected(monkeypatch):
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1))
    err, got = read_ptu("dummy.ptu", channel=-1)   # single-channel sentinel
    assert err == ""
    assert got.shape == (Y, X, H)
    np.testing.assert_array_equal(got, _expected(0))
