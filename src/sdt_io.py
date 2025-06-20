# -*- coding: utf-8 -*-
"""
read data from sdt file
"""
import sdtfile
from sdtfile import SdtFile
import numpy as np
import zipfile
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

def visualize_timebin(timeBin):
    """
    Visualize the timebin
    """
    import matplotlib.pyplot as plt
    import streamlit as st
    fig, ax = plt.subplots()
    plt.plot(timeBin)
    plt.show()
    st.pyplot(fig)  

def read_sdt_metadata(filename):
    
    with SdtFile(filename) as sdt:
        laser_rep_time = sdt.measure_info[0].rep_t
    return laser_rep_time   

def read_sdt_info_brukerSDT(filename):
    """ 
    modified from CGohlke sdtfile.py to read bruker 150 card data
    gives tarr, x.shape,y.shape,t.shape,c.shape
    """
    ## HEADER
    measure_info = []
    dtype = np.dtype(sdtfile.sdtfile.MEASURE_INFO)
    with open(filename, 'rb') as fh:
        ## HEADER
        header = np.rec.fromfile(fh, dtype=sdtfile.sdtfile.FILE_HEADER, shape=1, byteorder='<')
        fh.seek(header.meas_desc_block_offs[0])
        for _ in range(header.no_of_meas_desc_blocks[0]):
            measure_info.append(
                np.rec.fromfile(fh, dtype=dtype, shape=1, byteorder='<'))
            fh.seek(header.meas_desc_block_length[0] - dtype.itemsize, 1)
    
    times = []
    block_headers = []

    try:
        routing_channels_x = measure_info[0]['image_rx'][0]
    except:
        routing_channels_x = 1

    offset = header.data_block_offs[0]
 
    with open(filename, 'rb') as fh:
        for _ in range(header.no_of_data_blocks[0]): ## 
            fh.seek(offset)
            # read data block header
            bh = np.rec.fromfile(fh, dtype=sdtfile.sdtfile.BLOCK_HEADER, shape=1,
                                 byteorder='<')[0]
            block_headers.append(bh)
            # read data block
            mi = measure_info[bh.meas_desc_block_no]
            
            dtype = sdtfile.sdtfile.BlockType(bh.block_type).dtype
            dsize = bh.block_length // dtype.itemsize
            
            t = np.arange(mi.adc_re[0], dtype=np.float64)
            t *= mi.tac_r / float(mi.tac_g * mi.adc_re)
            times.append(t)
            offset = bh.next_block_offs
        return (header.data_block_offs[0], times, [mi.scan_x[0], mi.scan_y[0], mi.adc_re[0], routing_channels_x])
    
    
def read_sdt150(filename, channel=-1):
    """ sdt bruker uses data_block001 instead of data_block"""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    offset, t, XYTC = read_sdt_info_brukerSDT(filename)
    try: 
        # if the input file is sdt_zipped
        with zipfile.ZipFile(filename) as myzip:
            z1 = myzip.infolist()[0]  # "data_block"
            with myzip.open(z1.filename) as myfile:
                dataspl = myfile.read()
    except:
        # if the input file is unzipped 
        with open(filename, 'rb') as myfile:
            myfile.seek(offset)
            dataspl = myfile.read()
            
    dataSDT = np.fromstring(dataspl, np.uint16)

    if XYTC[3] == 1:
        # reduce the 4D data to 3d (CXYT to XYT)
        dataSDT = dataSDT[:XYTC[0] * XYTC[1] * XYTC[2]].reshape([XYTC[0], XYTC[1], XYTC[2]])
   
    elif XYTC[3] > 1:
        # Check for empty channels and filter them out
        actual_no_channels =  len(dataSDT) // (XYTC[0] * XYTC[1] * XYTC[2])
        if actual_no_channels == 1:
            # If only one channel is present, reshape to 3D
            dataSDT = dataSDT[:XYTC[0] * XYTC[1] * XYTC[2]].reshape([XYTC[0], XYTC[1], XYTC[2]])
        else:  # reshape XYTC to CXYT
            dataSDT = dataSDT[:XYTC[0] * XYTC[1] * XYTC[2] * actual_no_channels].reshape([actual_no_channels, XYTC[0], XYTC[1], XYTC[2]])
            if channel == -1: # return all channels
                return dataSDT
            try: 
                dataSDT = dataSDT[channel]
            except Exception as e:
                return None, f"Error reading sdt data: {e}"
    return dataSDT


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
    sdt_data = read_sdt150(src)
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