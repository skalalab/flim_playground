# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:59:27 2024
roi summing
@author: chris
@modified by Wenxuan to fit better with with flim-playground: 12/20/2024
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


def sum_sdts(images, write_tiff=False, write_sdt=False): 
    """
    Sum all the sdts with in a folder recursively
    """
    error_message = ""
    if len(images) == 0:
        error_message += "Either no sdt file found or no mask found associated with the sdt file! "
        return None, error_message
    for image, properties in images.items():
        mask = tiff.imread(Path(properties["mask"]))
        if "nadh_sdt" in properties:
            nadh_data = sdt_reader.read_sdt150(Path(properties["nadh_sdt"]))
            labels, summed_sdt, error_msg = sum_sdt(nadh_data, mask)
            error_message +=  error_msg
            properties["cells"] = labels
            properties["nadh_timebins"] = []
            for cell in labels: 
                properties["nadh_timebins"].append(summed_sdt[mask == cell][0])
 
        if "fad_sdt" in properties:
            fad_data = sdt_reader.read_sdt150(Path(properties["fad_sdt"]))
            labels, summed_sdt, error_msg = sum_sdt(fad_data, mask)
            error_message += error_msg
            properties["cells"] = labels
            properties["fad_timebins"] = []
            for cell in labels: 
                properties["fad_timebins"].append(summed_sdt[mask == cell][0])
    return images, error_message

def roi_sum_dimensionReduction(images, method="PCA", umap_neighbors=15, umap_min_dist=0.1):
    images, error_message = sum_sdts(images, write_tiff=False, write_sdt=False)
    if images is None:
        return None, error_message
    
    # step 1 : create a dataframe with all the time bins
    nadh_timebins = []
    fad_timebins = []
    nadh_cell_labels = []
    fad_cell_labels = []
    nadh_timebins_imageName = []
    fad_timebins_imageName = []
    nadh_categories = []
    fad_categories = []

    # from python 3.7, dictionaries maintain the insertion order of their keys
    for image, properties in images.items():
        if "nadh_timebins" in properties:
            # stack the nadh timebins for each image
            stacked_timebins = np.vstack(properties["nadh_timebins"])
            # we need to keep track of which image and which cell the timebin belongs to
            nadh_timebins_imageName.append(np.array([image]*stacked_timebins.shape[0]))
            nadh_cell_labels.append(properties["cells"])
            nadh_timebins.append(stacked_timebins)
            
            # use the parent folder name as the category
            nadh_parent = Path(properties["nadh_sdt"]).parent.name
            nadh_categories.append(np.array([nadh_parent]*stacked_timebins.shape[0]))
        if "fad_timebins" in properties:
            # stack the fad timebins for each image 
            stacked_timebins = np.vstack(properties["fad_timebins"])
            # we need to keep track of which image and which cell the timebin belongs to
            fad_timebins_imageName.append(np.array([image]*stacked_timebins.shape[0]))
            fad_cell_labels.append(properties["cells"])
            fad_timebins.append(stacked_timebins)  

            # use the parent folder name as the category
            fad_parent = Path(properties["fad_sdt"]).parent.name
            fad_categories.append(np.array([fad_parent]*stacked_timebins.shape[0]))

    nadh_timebins = np.vstack(nadh_timebins)
    fad_timebins = np.vstack(fad_timebins)
    nadh_cell_labels = np.hstack(nadh_cell_labels)
    fad_cell_labels = np.hstack(fad_cell_labels)
    nadh_timebins_imageName = np.hstack(nadh_timebins_imageName)
    fad_timebins_imageName = np.hstack(fad_timebins_imageName)
    nadh_categories = np.hstack(nadh_categories)
    fad_categories = np.hstack(fad_categories)

    # step 2: perform dimension reduction
    if nadh_timebins.size != 0:
        nadh_df, nadh_exp_var = dimension_reduction(nadh_timebins, n_components=2, method=method, umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist)
        # augment the dimensional reduction df with metadata
        nadh_df["image_name"] = nadh_timebins_imageName
        nadh_df["cell_labels"] = nadh_cell_labels
        nadh_df["base_name"] = nadh_df["image_name"] + "_" + nadh_df["cell_labels"].astype(str)
        nadh_df["color_category"] = nadh_categories
        nadh_df = fix_df(nadh_df)
    else: 
        nadh_df = None
        nadh_exp_var = None
    if fad_timebins.size != 0:
        fad_df, fad_exp_var = dimension_reduction(fad_timebins, n_components=2, method=method, umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist)
        # augment the dimensional reduction df with metadata
        fad_df["image_name"] = fad_timebins_imageName
        fad_df["cell_labels"] = fad_cell_labels
        fad_df["base_name"] = fad_df["image_name"] + "_" + fad_df["cell_labels"].astype(str)
        fad_df["color_category"] = fad_categories
        fad_df = fix_df(fad_df)
    else:
        fad_df = None
        fad_exp_var = None

    return nadh_df, nadh_exp_var, fad_df, fad_exp_var, error_message