# -*- coding: utf-8 -*-
"""
read data from raw decay file (sdt or ptu)
"""

from ptufile import PtuFile
from sdtfile import SdtFile
import numpy as np
from pathlib import Path
import os

def read_decay_metadata(filename):
    if not filename or not isinstance(filename, (str, os.PathLike)):
        return "Error: No decay file path provided.", None
    if not os.path.isfile(filename):
        return f"Error: Decay file not found: {filename}", None
    if filename.endswith(".ptu"):   
        try:
            ptu = PtuFile(filename)
        except Exception as e:
            return f"Error reading {filename}: file may be corrupted or truncated ({e})", None
        try:
            laser_rep_rate = ptu.tags['TTResult_SyncRate']
            if not laser_rep_rate or laser_rep_rate <= 0:
                return f"Error: Invalid laser sync rate ({laser_rep_rate}) in {filename}", None
            laser_rep_time = 1 / laser_rep_rate * 1e9
        except Exception:
            return f"Error: Cannot extract laser rep time from {filename}", None
    elif filename.endswith(".sdt"):
        try:
            sdt = SdtFile(filename)
        except Exception as e:
            return f"Error reading {filename}: file may be corrupted or truncated ({e})", None
        try:        
            tac_r = sdt.measure_info[0].tac_r
            tac_g = sdt.measure_info[0].tac_g
            if not tac_g or tac_g == 0:
                return f"Error: Invalid TAC gain (tac_g={tac_g}) in {filename}", None
            laser_rep_time = tac_r / tac_g * 1e9
        except Exception:
            return f"Error: Cannot extract laser rep time from {filename}", None
    else:
        return f"Error reading decay metadata: {filename} is not a valid sdt or ptu file", None
    return "", laser_rep_time   


def read_sdt(filename, channel=-1):
    try:
        sdt = SdtFile(filename)
    except Exception as e:
        return f"Error reading {filename}: file may be corrupted or truncated ({e})", None
    if len(sdt.data) == 0:
        return f"Error: {filename} contains no data blocks. The file may be empty or corrupted.", None
    if len(sdt.data) != 1:
        return f"Error: {filename} has {len(sdt.data)} data blocks (expected 1). It should be one field of view at a single time point (maybe multiple channels).", None
    else:
        # get the x, y, t, c
        try:
            x = int(sdt.measure_info[0].scan_x)
            y = int(sdt.measure_info[0].scan_y)
            t = int(sdt.measure_info[0].adc_re)
        except Exception:
            return f"Error: {filename} has no scan_x, scan_y, or adc_re", None
        if x <= 0 or y <= 0 or t <= 0:
            return f"Error: {filename} has invalid dimensions (x={x}, y={y}, t={t}). All must be > 0.", None
        decay_data = sdt.data[0]
        try: 
            c = sdt.measure_info[0].image_rx
        except:
            c = 1
        
        # checks if the data shape is consistent with the x, y, t, c
        shape_multiplier = x * y * t * c
        actual_shape_multiplier = np.prod(decay_data.shape)
        if shape_multiplier != actual_shape_multiplier:
            return f"Error: {filename} has inconsistent data shape with the metadata", None
        else:
            # reshpe the data to CYXT
            if c == 1:
                decay_data = decay_data.reshape(y, x, t)
            else:
                decay_data = decay_data.reshape(c, y, x, t)
        if channel != -1:
            num_channels = decay_data.shape[0] if c > 1 else 1
            if channel < 0 or channel >= num_channels:
                return f"Error: Channel index {channel} out of range for {filename} (available: 0-{num_channels - 1})", None
            decay_data = decay_data[channel]

        return "", decay_data

def read_ptu(filename, channel=-1):
    try:
        ptu = PtuFile(filename)
    except Exception as e:
        return f"Error reading {filename}: file may be corrupted or truncated ({e})", None
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
            return f"Error: {filename} has invalid dimensions (x={x}, y={y}, t={t}). All must be > 0.", None
        try:
            ptu_data = ptu[0]
        except Exception as e:
            return f"Error reading data from {filename}: {e}", None
        if c == 1:
            decay_data = ptu_data.reshape(y, x, t)
        else:
            decay_data = ptu_data.reshape(c, y, x, t)
        if channel != -1:
            num_channels = c
            if channel < 0 or channel >= num_channels:
                return f"Error: Channel index {channel} out of range for {filename} (available: 0-{num_channels - 1})", None
            decay_data = decay_data[channel]
        return "", decay_data
 
def read_decay(filename, channel=-1):
    if not filename or not isinstance(filename, (str, os.PathLike)):
        return "Error: No decay file path provided.", None
    if not os.path.isfile(filename):
        return f"Error: Decay file not found: {filename}", None
    if filename.endswith(".ptu"):
        error_msg, decay_data = read_ptu(filename, channel)
    elif filename.endswith(".sdt"):
        error_msg, decay_data = read_sdt(filename, channel)
    else:
        return f"Error reading decay data: {filename} is not a valid .sdt or .ptu file", None
    return error_msg, decay_data
