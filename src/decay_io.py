# -*- coding: utf-8 -*-
"""
read data from raw decay file (sdt or ptu)
"""

from ptufile import PtuFile
from sdtfile import SdtFile
import numpy as np
from pathlib import Path
import os

def visualize_sdt(sdt_data):
    """
    Visualize the sdt data
    """
    import matplotlib.pyplot as plt
    import streamlit as st
    fig, ax = plt.subplots()
    plt.imshow(sdt_data.sum(axis=-1))
    plt.show()
    st.pyplot(fig)

def read_decay_metadata(filename):
    if filename.endswith(".ptu"):   
        ptu = PtuFile(filename)
        try:
            laser_rep_rate = ptu.tags['TTResult_SyncRate']
            # convert to GHz and then to get the period in ns
            laser_rep_time = 1 / laser_rep_rate * 1e9
        except:
            return f"Error: Cannot extract laser rep time from {filename}", None
    elif filename.endswith(".sdt"):
        sdt = SdtFile(filename)
        try:        
            tac_r = sdt.measure_info[0].tac_r
            tac_g = sdt.measure_info[0].tac_g
            laser_rep_time = tac_r / tac_g * 1e9
        except:
            return f"Error: Cannot extract laser rep time from {filename}", None
    else:
        return f"Error reading decay metadata: {filename} is not a valid sdt or ptu file", None
    return "", laser_rep_time   


def read_sdt(filename, channel=-1):
    sdt = SdtFile(filename)
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
   
    ptu = PtuFile(filename)
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
        if c == 1:
            decay_data = ptu.reshape(y, x, t)
        else:
            decay_data = ptu.reshape(c, y, x, t)
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

def write_sdt(path_output, sdt_data, manufacturer="BH", resolution=256):
    

    # Requires the "sdtheader.dat" built header information
    
    ### Example1 : random dataset
    #binary_data=(np.random.randint(100,size=[256*256*256])).astype(np.uint16)
    
    ### Example2 : any data set with 256x256x256 - uint16
    # with open('badger.dat','rb') as fid:
    #     binary_data=np.fromstring(fid.read(),np.uint16)    
    
    #phantom_data= binary_data.ravel().astype(np.uint16)

    path_header = Path(f"./sdt_headers/header_{resolution}_{manufacturer}.dat")
    
    with open(path_header,'rb') as fid:
        header_ = fid.read() # prebuilt header_file 
    # combine header and data
    phantom_data = header_ + sdt_data.astype(np.uint16).tobytes()
    
    with open(path_output,'wb') as fid:
        fid.write(phantom_data)   


def sdt_convert(src, destination=""):
    """
    Convert a sdt file to a numpy array
    """
    errormsg = ""
    sdt_data = read_sdt(src)
    # the last dimension is time 
    if sdt_data.shape[-1] < 256:
        errormsg += f"{src} data should be at least 8-bit! "
        return errormsg
    elif sdt_data.shape[-1] == 256:
        errormsg += f"{src} data is already 8-bit! "
        return errormsg
    
    
    elif sdt_data.shape[-1] % 256 == 1:
        errormsg += f"{src} data is not divisible by 256! "
    else:
        #visualize_sdt(sdt_data)
        # Compute the grouping factor (how many neighbors to sum together)
        factor = sdt_data.shape[-1] // 256
       # visualize_timebin(sdt_data[0,0,:])
        # Reshape the last dimension to group neighbors
        new_shape = sdt_data.shape[:-1] + (256, factor)
        reshaped_data = sdt_data.reshape(new_shape)
        # Sum along the new axis
        time_grouped_data = reshaped_data.sum(axis=-1)
      #  visualize_timebin(time_grouped_data[0,0,:])
        #visualize_sdt(time_grouped_data)
       #  print(time_grouped_data.shape)
        write_sdt(os.path.join(destination, Path(src).name), time_grouped_data, resolution=sdt_data.shape[1], manufacturer="Swabian")


    return errormsg