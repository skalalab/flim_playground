"""Regression guard for the multi-channel PTU axis-scramble bug.

read_ptu took ptufile's (T,Y,X,C,H) buffer (ptu[0] -> (Y,X,C,H)) and then did
``reshape(c, y, x, t)``, which reinterprets the flat buffer as (C,Y,X,H) and
scrambles channels/pixels/time bins. Verified end-to-end on a real 2-channel
MicroTime 200 beads.ptu: channel photon totals 26870/10514 became 35107/2277.
Fixed with ``np.moveaxis(ptu_data, c_axis, 0)``. (read_sdt's identical reshape is
safe only because sdtfile already returns channel-first (C,Y,X,T).)

ptufile is mocked so this needs no binary fixture: a coordinate-encoded
(T,Y,X,C,H) buffer stores each voxel's own (c,y,x,h); read_ptu must return every
channel un-scrambled.
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

    # Guard: the OLD blind reshape really would have scrambled this buffer, so the
    # assertions above are a meaningful test (not trivially satisfiable).
    buf0 = _FakePtu(2)[0]                 # (Y, X, C, H)
    old = buf0.reshape(2, Y, X, H)[0]     # the buggy operation
    assert not np.array_equal(old, _expected(0)), "test dims too small to expose the bug"


def test_read_ptu_single_channel_unaffected(monkeypatch):
    monkeypatch.setattr(decay_io, "PtuFile", lambda _fn: _FakePtu(1))
    err, got = read_ptu("dummy.ptu", channel=-1)   # single-channel sentinel
    assert err == ""
    assert got.shape == (Y, X, H)
    np.testing.assert_array_equal(got, _expected(0))
