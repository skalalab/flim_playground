# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:59:27 2024
roi summing
@author: Wenxuan Zhao
@Adopted from Chris' and Kayvan's roi summing code to fit better with with flim-playground
"""

import tifffile as tiff
import numpy as np
import sdt_reader
import os, shutil
import pandas as pd
from pathlib import Path
from dimension_reduction import dimension_reduction
from features import fix_df

#%%
def sum_sdt(sdt_data, mask):
    """
    sdt can be 3 or 4 dimensional , mask is 2 dimension. Output is 3 dimensional
    """
    error_msg = ""
    
    if len(mask.shape) != 2:
        error_msg += "Your Mask should be 2 dimension !"
        return None, None, error_msg
    
    if len(sdt_data.shape) == 3:
        # sdt_data is XYT
        channel_data = sdt_data
    elif len(sdt_data.shape) == 4:
        # sdt_data is CXYT
        for channel in range(sdt_data.shape[0]):
            if (np.count_nonzero(sdt_data[channel]) == 0):
                continue
            
            else: 
                channel_data = sdt_data[channel]
                break
    else:
        error_msg += "Your sdt data should be either 3 or 4 dimensions! "
        return None, None, error_msg
        
    cell_labels = np.unique(mask)
    cell_labels = cell_labels[cell_labels != 0]
    
    summed_sdt = np.zeros_like(channel_data)
    for label in cell_labels:
        # create a binary mask for the current cell label
        label_mask = (mask == label)

        # filter out the subset of sdt data that have coordinates (x, y) in where the mask[x,y] = label
        # channel_data[label_mask, :] will collapse the first two dimension into 1. 
        # To sum the timebins, we use .sum(axis=0) to collapse that dimension again
        time_sum = channel_data[label_mask, :].sum(axis=0)

        # Assign the summed values to the corresponding pixels in new_data
        summed_sdt[label_mask, :] = time_sum

    return cell_labels, summed_sdt, error_msg


def sum_sdts(images, selected_channel="NADH", write_tiff=False, write_sdt=False): 
    """
    Sum all the sdts with in a folder recursively
    """
    error_message = ""
    if len(images) == 0:
        error_message += "Either no sdt file found or no mask found associated with the sdt file! "
        return None, error_message
    
    timeBin_name = f"{selected_channel.lower()}_timebins"
    sdt_Path = f"{selected_channel.lower()}_sdt"
    for image, properties in images.items():
        if "mask" in properties and sdt_Path in properties:
            mask = tiff.imread(Path(properties["mask"])) 
            sdt_data = sdt_reader.read_sdt150(Path(properties[sdt_Path]))
            labels, summed_sdt, error_msg = sum_sdt(sdt_data, mask)
            error_message +=  error_msg
            properties["cells"] = labels
            properties[timeBin_name] = []
            for cell in labels: 
                properties[timeBin_name].append(summed_sdt[mask == cell][0])
 
    return images, error_message

def roi_sum_dimensionReduction(images, selected_channel="NADH", method="PCA", umap_neighbors=15, umap_min_dist=0.1):
    """
    method = "PCA" or "UMAP"

    """

    images, error_message = sum_sdts(images,selected_channel=selected_channel, write_tiff=False, write_sdt=False)
    if images is None:
        return None, None, error_message
         
    # step 1 : create a dataframe with all the time bins
    timebins = []
    cell_labels = []
    timebins_imageName = []
    categories = []

    # selected_channel.lower() = nadh or fad
    timeBin_name = f"{selected_channel.lower()}_timebins"
    sdt_Path = f"{selected_channel.lower()}_sdt"
    # from python 3.7, dictionaries maintain the insertion order of their keys
    for image, properties in images.items():
        if timeBin_name in properties:
            # stack the nadh timebins for each image
            stacked_timebins = np.vstack(properties[timeBin_name])
            # we need to keep track of which image and which cell the timebin belongs to
            timebins_imageName.append(np.array([image]*stacked_timebins.shape[0]))
            cell_labels.append(properties["cells"])
            timebins.append(stacked_timebins)
            
            # use the parent folder name as the category
            image_parent = Path(properties[sdt_Path]).parent.name
            categories.append(np.array([image_parent]*stacked_timebins.shape[0]))

    timebins = np.vstack(timebins)
    cell_labels = np.hstack(cell_labels)
    timebins_imageName = np.hstack(timebins_imageName)
    categories = np.hstack(categories)

    # step 2: perform dimension reduction
    if timebins.size != 0:
        df, exp_var = dimension_reduction(timebins, n_components=2, method=method, umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist)
        # augment the dimensional reduction df with metadata
        df["image_name"] = timebins_imageName
        df["cell_labels"] = cell_labels
        df["base_name"] = df["image_name"] + "_" + df["cell_labels"].astype(str)
        df["color_category"] = categories
        df = fix_df(df)
    else: 
        df = None
        exp_var = None

    return df, exp_var, error_message


