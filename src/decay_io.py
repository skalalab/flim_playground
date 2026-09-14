"""Read raw decay data and acquisition metadata from SDT or PTU files.

PTU histograms are sized by the laser period: time bins = period / TCSPC
resolution, both read from the file's tags, never the highest bin that happened
to receive a photon. Every field of view from one acquisition therefore shares
one shape. The ``duration`` that ``read_decay_metadata`` reports is time bins x
resolution, the window those bins span, so the bin width downstream is the
file's own TCSPC resolution; it is not 1 / sync rate, which Leica FALCON exports
round to a nominal value. Repeated frames within one file are summed into a
single decay image.
"""

import math
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from ptufile import PtuFile
from sdtfile import SdtFile


def _msg_no_path():
    return "Error: No decay file path provided."


def _msg_not_found(filename):
    return f"Error: Decay file not found: {filename}"


def _msg_corrupted(filename, e):
    return f"Error reading {filename}: file may be corrupted or truncated ({e})"


def _msg_no_laser_rep(filename):
    return f"Error: Cannot extract laser rep time from {filename}"


def _msg_no_time_axis(filename, e):
    return (
        f"Error: cannot determine the time bins of {filename} from its laser "
        f"sync rate and TCSPC resolution ({e})"
    )


def _ptu_time_axis(ptu):
    """Return ``(time_bins, duration_ns)`` for a T3 imaging PTU file.

    ``time_bins`` is floor(laser period / TCSPC resolution): the bins that fit in
    one sync period, fixed by the instrument and so shared by every FOV.
    ``duration_ns`` is time_bins x resolution, the window those bins span, so that
    ``duration / time_bins`` downstream is the file's TCSPC resolution. It is not
    1 / sync rate: Leica FALCON exports write a rounded nominal rate (80 MHz for a
    laser near 79.5 MHz), and dividing that by the bin count stretches every bin
    by the period's fractional bin (0.7 % on those files).
    """
    period = 1 / ptu.tags["TTResult_SyncRate"]
    resolution = ptu.tcspc_resolution
    time_bins = math.floor(period / resolution)
    return time_bins, float(time_bins * resolution * 1e9)


def _msg_channel_oob(channel, filename, num_channels):
    return f"Error: Channel index {channel} out of range for {filename} (available: 0-{num_channels - 1})"


def _msg_bad_dims(filename, x, y, t):
    return f"Error: {filename} has invalid dimensions (x={x}, y={y}, t={t}). All must be > 0."


def _validate_decay_path(filename):
    """Shared path validation for decay readers. Returns "" if OK, else an error string."""
    if not filename or not isinstance(filename, (str, os.PathLike)):
        return _msg_no_path()
    if not os.path.isfile(filename):
        return _msg_not_found(filename)
    return ""


def read_decay_metadata(filename):
    err = _validate_decay_path(filename)
    if err:
        return err, None
    if filename.endswith(".ptu"):
        try:
            ptu = PtuFile(filename)
        except Exception as e:
            return _msg_corrupted(filename, e), None
        try:
            laser_rep_rate = ptu.tags['TTResult_SyncRate']
            if not laser_rep_rate or laser_rep_rate <= 0:
                return f"Error: Invalid laser sync rate ({laser_rep_rate}) in {filename}", None
        except Exception:
            return _msg_no_laser_rep(filename), None
        try:
            # The window the period bins span, not 1 / sync rate: see _ptu_time_axis.
            _, laser_rep_time = _ptu_time_axis(ptu)
        except Exception as e:
            return _msg_no_time_axis(filename, e), None
    elif filename.endswith(".sdt"):
        try:
            sdt = SdtFile(filename)
        except Exception as e:
            return _msg_corrupted(filename, e), None
        try:
            tac_r = sdt.measure_info[0].tac_r
            tac_g = sdt.measure_info[0].tac_g
            if not tac_g or tac_g == 0:
                return f"Error: Invalid TAC gain (tac_g={tac_g}) in {filename}", None
            # Convert SDT float32 fields to a native float for lmfit JSON serialization.
            laser_rep_time = float(tac_r / tac_g * 1e9)
        except Exception:
            return _msg_no_laser_rep(filename), None
    else:
        return f"Error reading decay metadata: {filename} is not a valid sdt or ptu file", None
    return "", laser_rep_time


def read_sdt(filename, channel=-1):
    try:
        sdt = SdtFile(filename)
    except Exception as e:
        return _msg_corrupted(filename, e), None

    if len(sdt.data) == 0:
        return f"Error: {filename} contains no data blocks. The file may be empty or corrupted.", None
    if len(sdt.data) != 1:
        return (
            f"Error: {filename} has {len(sdt.data)} data blocks (expected 1). "
            f"It should be one field of view at a single time point "
            f"(maybe multiple channels)."
        ), None

    try:
        x = int(sdt.measure_info[0].scan_x)
        y = int(sdt.measure_info[0].scan_y)
        t = int(sdt.measure_info[0].adc_re)
    except Exception:
        return f"Error: {filename} has no scan_x, scan_y, or adc_re", None
    if x <= 0 or y <= 0 or t <= 0:
        return _msg_bad_dims(filename, x, y, t), None

    decay_data = sdt.data[0]
    try:
        c = int(sdt.measure_info[0].image_rx)
    except Exception:
        c = 1

    shape_multiplier = x * y * t * c
    actual_shape_multiplier = np.prod(decay_data.shape)
    if shape_multiplier != actual_shape_multiplier:
        return (
            f"Error: {filename} has inconsistent data shape with the metadata "
            f"(expected {shape_multiplier}, got {actual_shape_multiplier})"
        ), None

    if c == 1:
        decay_data = decay_data.reshape(y, x, t)
    else:
        decay_data = decay_data.reshape(c, y, x, t)

    if channel != -1:
        num_channels = decay_data.shape[0] if c > 1 else 1
        if channel < 0 or channel >= num_channels:
            return _msg_channel_oob(channel, filename, num_channels), None
        decay_data = decay_data[channel]

    return "", decay_data


def _decode_ptu(filename, channel=-1):
    """Decode a T3 imaging PTU file into ``(error_msg, decay_data, n_frames)``.

    Frames are integrated inside the decoder and the time axis is cut (or padded)
    to the laser period, so the bins span the ``duration`` that
    ``read_decay_metadata`` derives from the same two tags and every FOV of an
    acquisition decodes to one shape.
    """
    try:
        ptu = PtuFile(filename)
    except Exception as e:
        return _msg_corrupted(filename, e), None, 0
    with ptu:  # release the records buffer (GBs for long acquisitions) on exit
        try:
            n_frames = int(ptu.shape[ptu.dims.index("T")])
            c = ptu.shape[ptu.dims.index("C")]
            y = ptu.shape[ptu.dims.index("Y")]
            x = ptu.shape[ptu.dims.index("X")]
        except Exception:
            return f"Error: {filename} has no T, C, Y, or X dimension", None, 0
        try:
            t, _ = _ptu_time_axis(ptu)
        except Exception as e:
            return _msg_no_time_axis(filename, e), None, 0
        if x <= 0 or y <= 0 or t <= 0:
            return _msg_bad_dims(filename, x, y, t), None, 0
        try:
            # frame=-1 sums the frame axis in the decoder; 32-bit bins keep the sum from wrapping.
            ptu_data = ptu.decode_image(frame=-1, dtime=t, dtype=np.uint32)[0]
        except Exception as e:
            return f"Error reading data from {filename}: {e}", None, 0
        if c == 1:
            decay_data = ptu_data.reshape(y, x, t)
        else:
            # Reorder (Y, X, C, H) to (C, Y, X, H) for channel indexing.
            c_axis = ptu.dims.index("C") - 1  # -1: [0] dropped the leading T axis
            decay_data = np.moveaxis(ptu_data, c_axis, 0)
    if channel != -1:
        if channel < 0 or channel >= c:
            return _msg_channel_oob(channel, filename, c), None, 0
        decay_data = decay_data[channel]
    return "", decay_data, n_frames


def read_ptu(filename, channel=-1):
    error_msg, decay_data, _ = _decode_ptu(filename, channel)
    return error_msg, decay_data


@dataclass(frozen=True)
class PtuTiming:
    """Acquisition timing from header tags, without scanning photon records."""

    time_bins: int
    duration: float  # ns
    laser_rate: float  # GHz


def read_ptu_reference_timing(filename):
    """Validate a reference's header and return its timing without decoding it.

    Channel count and nonempty photon histograms are checked by the full
    reference reader when calibration or extraction actually needs the curve.
    """
    err = _validate_decay_path(filename)
    if err:
        return err, None
    try:
        with PtuFile(filename) as ptu:
            if not ptu.is_t3 or not ptu.is_image:
                return f"Error: PTU reference {filename} must contain T3 imaging data.", None
            if ptu.number_records == 0:
                return f"Error: PTU reference {filename} is empty.", None
            try:
                time_bins, duration = _ptu_time_axis(ptu)
                laser_rate = float(ptu.tags["TTResult_SyncRate"] * 1e-9)
                if (time_bins <= 0 or not math.isfinite(duration) or duration <= 0
                        or not math.isfinite(laser_rate) or laser_rate <= 0):
                    raise ValueError("acquisition timing must be positive and finite")
            except Exception as e:
                return _msg_no_time_axis(filename, e), None
            return "", PtuTiming(time_bins, duration, laser_rate)
    except Exception as e:
        return _msg_corrupted(filename, e), None


@dataclass(frozen=True)
class PtuReference:
    """Integrated reference counts; duration is in ns and laser_rate in GHz."""

    curve: np.ndarray
    time_bins: int
    duration: float
    laser_rate: float
    n_frames: int


@lru_cache(maxsize=32)
def _read_ptu_reference_cached(filename, file_size, mtime_ns):
    """Cache only the compact histogram, keyed by the file's identity on disk."""
    try:
        with PtuFile(filename) as ptu:
            if not ptu.is_t3 or not ptu.is_image:
                return f"Error: PTU reference {filename} must contain T3 imaging data.", None
            if ptu.number_records == 0:
                return f"Error: PTU reference {filename} is empty.", None
            # Cache the read-only mapping before inspecting shape: ptufile scans
            # records to discover frames/channels and otherwise reads them into RAM.
            records = ptu.read_records(memmap=True, cache=True)
            sizes = dict(zip(ptu.dims, ptu.shape))
            if not all(dim in sizes for dim in ("T", "Y", "X", "C", "H")):
                return f"Error: PTU reference {filename} must contain T3 imaging data.", None
            if sizes["C"] != 1:
                return (f"Error: PTU reference {filename} must be single-channel; "
                        f"found {sizes['C']} channels."), None
            try:
                time_bins, duration = _ptu_time_axis(ptu)
            except Exception as e:
                return _msg_no_time_axis(filename, e), None
            if min(sizes["T"], sizes["Y"], sizes["X"], time_bins) <= 0:
                return _msg_bad_dims(filename, sizes["X"], sizes["Y"], time_bins), None
            # A negative slice step integrates the selected axis in the decoder.
            # No (frames, pixels, bins) image is allocated, even for long scans.
            selection = tuple(slice(None, None, -1) if dim in ("T", "Y", "X")
                              else slice(None) for dim in ptu.dims)
            curve = ptu.decode_image(selection, records=records, dtime=time_bins,
                                     dtype=np.uint64).reshape(-1)
            if not np.any(curve):
                return f"Error: PTU reference {filename} is empty (no decoded photons in the acquisition window).", None
            return "", PtuReference(curve, time_bins, duration,
                                    float(ptu.tags["TTResult_SyncRate"] * 1e-9),
                                    int(sizes["T"]))
    except Exception as e:
        return _msg_corrupted(filename, e), None


def read_ptu_reference(filename):
    """Return ``(error_msg, PtuReference)`` for a single-channel imaging PTU.

    All decoded pixels and repeated frames contribute within the same
    acquisition window as the sample reader. Callers receive their own curve
    so background subtraction or IRF processing cannot mutate the cached data.
    """
    err = _validate_decay_path(filename)
    if err:
        return err, None
    try:
        path = Path(filename).resolve()
        stat = path.stat()
        err, reference = _read_ptu_reference_cached(str(path), stat.st_size, stat.st_mtime_ns)
    except OSError as e:
        return _msg_corrupted(filename, e), None
    if err:
        return err, None
    return "", replace(reference, curve=reference.curve.copy())


def clear_ptu_reference_cache():
    """Discard compact references when the user requests a folder rescan."""
    _read_ptu_reference_cached.cache_clear()


def read_decay_with_frames(filename, channel=-1):
    """Return ``(error_msg, (decay_data, n_frames))``.

    ``n_frames`` is how many repeated frames were summed into ``decay_data``;
    it is always 1 for SDT files.
    """
    err = _validate_decay_path(filename)
    if err:
        return err, None
    if filename.endswith(".ptu"):
        error_msg, decay_data, n_frames = _decode_ptu(filename, channel)
    elif filename.endswith(".sdt"):
        error_msg, decay_data = read_sdt(filename, channel)
        n_frames = 1
    else:
        return f"Error reading decay data: {filename} is not a valid .sdt or .ptu file", None
    if error_msg != "":
        return error_msg, None
    return "", (decay_data, n_frames)


def read_decay(filename, channel=-1):
    error_msg, result = read_decay_with_frames(filename, channel)
    if error_msg != "":
        return error_msg, None
    return "", result[0]
