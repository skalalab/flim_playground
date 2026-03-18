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
    if filename.endswith(".ptu"):   
        try:
            ptu = PtuFile(filename)
        except Exception as e:
            return f"Error reading {filename}: file may be corrupted or truncated ({e})", None
        try:
            laser_rep_rate = ptu.tags['TTResult_SyncRate']
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
    if len(sdt.data) != 1:
        return f"Error: {filename} has multiple time frames. It should be on one field of view at a single time point (maybe multiple channels).", None
    else:
        # get the x, y, t, c
        try:
            x = sdt.measure_info[0].scan_x
            y = sdt.measure_info[0].scan_y
            t = sdt.measure_info[0].adc_re   
        except:
            return f"Error: {filename} has no scan_x, scan_y, or adc_re", None
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
        except:
            return f"Error: {filename} has no C, Y, X, or H dimension", None
        ptu_data = ptu[0]
        if c == 1:
            decay_data = ptu_data.reshape(y, x, t)
        else:
            decay_data = ptu_data.reshape(c, y, x, t)
        if channel != -1:
            decay_data = decay_data[channel]
        return "", decay_data
 
def read_decay(filename, channel=-1):
    if filename.endswith(".ptu"):
        error_msg, decay_data = read_ptu(filename, channel)
    elif filename.endswith(".sdt"):
        error_msg, decay_data = read_sdt(filename, channel)
    else:
        return None, f"Error reading decay data: {filename} is not a valid sdt or ptu file"
    return error_msg, decay_data
