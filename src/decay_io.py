"""
read data from raw decay file (sdt or ptu)
"""

import os

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
            laser_rep_time = float(1 / laser_rep_rate * 1e9)
        except Exception:
            return _msg_no_laser_rep(filename), None
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
            # tac_r / tac_g are float32 SDT header fields; coerce so `duration`
            # stays a native float (float32 is not JSON-serializable, which
            # otherwise breaks lmfit params.dumps() on the parallel fit path).
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

def read_ptu(filename, channel=-1):
    try:
        ptu = PtuFile(filename)
    except Exception as e:
        return _msg_corrupted(filename, e), None
    if ptu.shape[0] != 1:
        return f"Error: {filename} has multiple time frames. It should be on one field of view at a single time point (maybe multiple channels).", None
    else:
        try: 
            c = ptu.shape[ptu.dims.index("C")]
            y = ptu.shape[ptu.dims.index("Y")]
            x = ptu.shape[ptu.dims.index("X")]
            t = ptu.shape[ptu.dims.index("H")]
        except Exception:
            return f"Error: {filename} has no C, Y, X, or H dimension", None
        if x <= 0 or y <= 0 or t <= 0:
            return _msg_bad_dims(filename, x, y, t), None
        try:
            ptu_data = ptu[0]
        except Exception as e:
            return f"Error reading data from {filename}: {e}", None
        if c == 1:
            decay_data = ptu_data.reshape(y, x, t)
        else:
            # ptu[0] is laid out as (Y, X, C, H) (ptu.dims minus the leading T).
            # Move the channel axis to the front -> (C, Y, X, H) so the later
            # decay_data[channel] actually selects a channel. A blind
            # reshape(c, y, x, t) reinterprets this buffer and scrambles
            # channels/pixels/time bins. (read_sdt's identical reshape is safe
            # only because sdtfile already returns channel-first (C, Y, X, T).)
            c_axis = ptu.dims.index("C") - 1  # -1: ptu[0] dropped the leading T axis
            decay_data = np.moveaxis(ptu_data, c_axis, 0)
        if channel != -1:
            num_channels = c
            if channel < 0 or channel >= num_channels:
                return _msg_channel_oob(channel, filename, num_channels), None
            decay_data = decay_data[channel]
        return "", decay_data
 
def read_decay(filename, channel=-1):
    err = _validate_decay_path(filename)
    if err:
        return err, None
    if filename.endswith(".ptu"):
        error_msg, decay_data = read_ptu(filename, channel)
    elif filename.endswith(".sdt"):
        error_msg, decay_data = read_sdt(filename, channel)
    else:
        return f"Error reading decay data: {filename} is not a valid .sdt or .ptu file", None
    return error_msg, decay_data
