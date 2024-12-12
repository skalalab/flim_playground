# -*- coding: utf-8 -*-
"""
read data from sdt file
"""
import sdtfile
import numpy as np
import zipfile

def read_sdt_info_brukerSDT(filename):
    """ 
    modified from CGohlke sdtfile.py to read bruker 150 card data
    gives tarr, x.shape,y.shape,t.shape,c.shape
    """
    ## HEADER
    with open(filename, 'rb') as fh:
        header = np.rec.fromfile(fh, dtype=sdtfile.sdtfile.FILE_HEADER, shape=1, byteorder='<')
        
    measure_info = []
    dtype = np.dtype(sdtfile.sdtfile.MEASURE_INFO)
    with open(filename, 'rb') as fh:
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

    print()
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
        return (times, [mi.scan_x[0], mi.scan_y[0], mi.adc_re[0], routing_channels_x])
    
    
def read_sdt150(filename):
    """ sdt bruker uses data_block001 instead of data_block"""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    t, XYTC = read_sdt_info_brukerSDT(filename)
    
    with zipfile.ZipFile(filename) as myzip:
        z1 = myzip.infolist()[0]  # "data_block"
        with myzip.open(z1.filename) as myfile:
            dataspl = myfile.read()
            
    dataSDT = np.fromstring(dataspl, np.uint16)

    if XYTC[3] > 1:
        dataSDT = dataSDT[:XYTC[0] * XYTC[1] * XYTC[2] * XYTC[3]].reshape([XYTC[3], XYTC[0], XYTC[1], XYTC[2]])
        # if dataSDT[0, :, :, :].sum() == 0:  # bruker uses two channels and keeps one empty!!!
        #     dataSDT = np.squeeze(dataSDT[1, :, :, :])
        # else:
        #     if dataSDT[1, :, :, :].sum() == 0:  # bruker uses two channels and keeps one empty!!!
        #         dataSDT = np.squeeze(dataSDT[0, :, :, :])
        #         # print ' ch1 is empty'
        #     else:
        #         pass
    else:
        dataSDT = dataSDT[:XYTC[0] * XYTC[1] * XYTC[2] * XYTC[3]].reshape([XYTC[0], XYTC[1], XYTC[2]])
    # print("READ DATA IN:",dataSDT.shape)
    return (dataSDT)