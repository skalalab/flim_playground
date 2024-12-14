# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:59:27 2024

roi summing

@author: chris
"""

import tifffile as tiff
import numpy as np
import sdt_reader
import os, shutil
from pathlib import Path

# roi sum
#
# param: sdt
# param: mask
def sum_roi(sdt, mask_path):
    
    image_name = sdt.name[:sdt.name.find(".sdt")]
    
    if not os.path.exists("Summed"):
        os.mkdir("Summed")
    
    # get image datafrom
    sdt_data = sdt_reader.read_sdt150(sdt)
        
    # sum each nonempty channel
    for channel in range(sdt_data.shape[0]):
        if (np.count_nonzero(sdt_data[channel]) == 0):
            continue
        
        channel_data = sdt_data[channel]

        # now mask the image
        with tiff.TiffFile(mask_path) as mask_tif:
            mask = mask_tif.asarray()
            
        masked_image = np.copy(channel_data)

        # apply mask to sdt!
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row][col] == 0:
                    masked_image[row][col][:] = 0
        
        # get cell values
        cell_values = np.unique(mask)
        cell_values = cell_values[1:]
    
        # single out each cell
        cell_sums = list()
        for i in range(len(cell_values)):
            # mask each cell
            cell_image = np.copy(masked_image)
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row][col] != cell_values[i]:
                        cell_image[row][col][:] = 0
                            
            # sum each cell
            cell_hist = np.sum(cell_image, 0)
            cell_hist = np.sum(cell_hist, 0)
            cell_sums.append(cell_hist)
           
        # make summed roi
        summed_image = np.copy(masked_image)
        for i in range(len(cell_values)):
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row][col] == cell_values[i]:
                        np.put(summed_image[row][col], range(summed_image.shape[2]), cell_sums[i])
        
        summed_image = np.swapaxes(summed_image, 0, 2)
        summed_image = np.swapaxes(summed_image, 1, 2)
        
        # save file
        tiff.imwrite("Summed/" + image_name + "_channel" + str(channel) + "_summed.tif", summed_image)
           

# test sum_roi
#
# return: true if gud
def test_sum_roi():
    if os.path.exists("Summed"):
        shutil.rmtree("Summed")
    
    # test 1
    sum_roi(Path("dHL60_Control_DMSO_02_n-024.sdt"), Path("dHL60_Control_DMSO_02_n-024_photons_cellpose.tiff"))
    
    summed_path = "Summed/dHL60_Control_DMSO_02_n-024_channel1_summed.tif"
    
    if not os.path.exists(summed_path):
        return False
    
    # get summed data (actual)
    with tiff.TiffFile(summed_path) as summed_tif:    
        summed_image = summed_tif.asarray()
    
    # get sdt data (expected)
    sdt_data = sdt_reader.read_sdt150("dHL60_Control_DMSO_02_n-024_summed.sdt")
    sdt_data = sdt_data[1]
    sdt_data = np.swapaxes(sdt_data, 0, 2)
    sdt_data = np.swapaxes(sdt_data, 1, 2)
    
    if not np.array_equal(summed_image, sdt_data):
        return False
    
    # test 2
    sum_roi(Path("dHL60_Control_na_01_n-010.sdt"), Path("dHL60_Control_na_01_n-010_photons_cellpose.tiff"))
    
    summed_path = "Summed/dHL60_Control_na_01_n-010_channel1_summed.tif"
    
    if not os.path.exists(summed_path):
        return False
    
    # get summed data (actual)
    with tiff.TiffFile(summed_path) as summed_tif:    
        summed_image = summed_tif.asarray()
    
    # get sdt data (expected)
    sdt_data = sdt_reader.read_sdt150("dHL60_Control_na_01_n-010_summed.sdt")
    sdt_data = sdt_data[1]
    sdt_data = np.swapaxes(sdt_data, 0, 2)
    sdt_data = np.swapaxes(sdt_data, 1, 2)
    
    if not np.array_equal(summed_image, sdt_data):
        return False
    
    
    # all good
    return True


# test
print(test_sum_roi())  
sum_roi(Path("C:/Users/chris/flute_pipeline/fish-050.sdt"), Path("C:/Users/chris/flute_pipeline/fish-050.tiff"))