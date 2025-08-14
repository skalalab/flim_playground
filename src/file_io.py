import codecs
import numpy as np
import pathlib
from pathlib import Path
import tifffile
from typing import Union
from src.decay_io import read_decay
from src.config import get_fov_name_col
import pandas as pd

def load_image(path: Union[str, pathlib.PurePath]) -> np.ndarray:
    """
    Detects the extension and loads image into a numpy array 
    if it's a tif/tiff or an asc file.

    Parameters
    ----------
    path : pathlib path or str
        path to the image.

    Returns
    -------
    np.ndarray
        ndarray with the image data.

    """
    if not isinstance(path, pathlib.PurePath):
        path = Path(path)
    pass
    if path.suffix == ".asc":
        return read_asc(path)
    if path.suffix in [".tiff", ".tif"]:
        return tifffile.imread(path)
    
def read_asc(path):
    """
    Reads in an asc file into a numpy ndarray

    Parameters
    ----------
    path : pathlib.Path
        path to the file.

    Returns
    -------
    array : np.ndarray
        Numpy array holding the image data.

    """
    with codecs.open(path, encoding="utf-8-sig") as file:
        # for each line for each value, convert to float
        array = np.array([[float(x) for x in line.split()] for line in file])

    return array

def _get_sample_decay_curves(decays: pd.DataFrame, n_samples: int, max_intensity: float):
    # use the top n_samples that have the highest intensity less than max_intensity
    # return the decay curves
    decay_intensity = np.sum(decays, axis=1)
    sorted_indices = np.argsort(decay_intensity)[::-1]
    filtered_indices = [idx for idx in sorted_indices if decay_intensity[idx] < max_intensity]

    if len(filtered_indices) < n_samples:
        top_indices = filtered_indices
    else:
        top_indices = filtered_indices[:n_samples]
    return decays.iloc[top_indices].values, top_indices

def get_decay_curves(metadata_df, input_type, channel_name, time_bins, shift=True):

    # step 1 file check
    # decay file, irf file, mask file for roi summing and SPCImage
    # histogram, irf for K-Flow
    error_msg = ""
    decay_curves = {}
    
    # Handle both DataFrame and Series cases

    if isinstance(metadata_df, pd.Series):
        # metadata_df is already a single row (Series)
        first_row = metadata_df
        num_fovs = 1
    else:
        # metadata_df is a DataFrame, get the first row
        first_row = metadata_df.iloc[0]
        num_fovs = len(metadata_df)
    
    irf_path = first_row.get(f'{channel_name}_IRF', None)
    fov_name_col = get_fov_name_col()
    try:
        irf = np.loadtxt(irf_path)
    except Exception as e:
        return f"Error: IRF file not found for {channel_name}.", None, None
    if len(irf) != time_bins:
        return f"IRF length mismatch with specified time bins. IRF length: {len(irf)}, time bins: {time_bins}.", None, None
    # Handle iteration for both DataFrame and Series cases
    if isinstance(metadata_df, pd.Series):
        # Process single row (Series)
        rows_to_process = [(0, metadata_df)]
    else:
        # Process DataFrame rows
        rows_to_process = list(metadata_df.iterrows())
    
    for i, row in rows_to_process:
        fov_name = row[fov_name_col]
        decay_path = row.get(f'{channel_name}_Decay', None)
        if "Decay (3/4D)" in input_type:
            mask_path = row.get(f'{channel_name}_Mask', None)
            try:
                mask = load_image(mask_path)
            except Exception as e:
                return f"Error reading the mask file for {fov_name_col} {fov_name} at {mask_path}: {e}", None, None  
            channel_no = row.get(f'{channel_name}_channel', None)
            if channel_no is None:
                return f"Error: Channel number not found for {fov_name_col} {fov_name}", None, None      
            try:
                error_msg, decay = read_decay(decay_path, channel_no)
                if error_msg != "":
                    return error_msg, None, None
            except Exception as e:
                return f"Error reading the decay file for {fov_name_col} {fov_name} at {decay_path}: {e}", None, None
            if len(decay.shape) != 3:
                return f"Decay data mismatch. Expected 3D data (XYT), got {decay.shape}.", None, None
            if decay.shape[-1] != time_bins:
                return f"Decay time bins mismatch. Decay time bins: {decay.shape[2]}, time bins: {time_bins}.", None, None
            if len(mask.shape) != 2:
                return f"Mask data mismatch. Expected 2D data, got {mask.shape}.", None, None
            if decay.shape[0] != mask.shape[0] or decay.shape[1] != mask.shape[1]:
                return f"Dimension mismatch: Decay data {decay.shape[:2]} vs mask {mask.shape}", None, None
            
            if shift:
                # binarize the mask
                binary_mask = np.where(mask > 0, 1, 0)
                # image_level ROI summing: sum the time axis of all non-zero pixels in the decay
                summed_decay_curve = np.sum(decay * binary_mask[:, :, np.newaxis], axis=(0, 1))
                decay_curves[fov_name] = summed_decay_curve
            else:
                # get all the decay curves for each cell
                unique_cells = np.sort(np.unique(mask))
                # Remove background (0)
                unique_cells = unique_cells[unique_cells != 0]
                for cell in unique_cells:
                    cell_mask = mask == cell
                    cell_id = f"{fov_name}_{cell}"
                    decay_curves[cell_id] = decay[cell_mask, :].sum(axis=0)

        elif input_type == "Decay (2D)":
            try:
                decays = pd.read_csv(decay_path)    
            except Exception as e:
                return f"Error reading the histogram file for {fov_name_col} {fov_name} at {decay_path}: {e}", None, None
            if len(decays.shape) != 2:
                return f"Decay data mismatch. Expected 2D data, got {decays.shape}.", None, None
            if decays.shape[1] != time_bins:
                return f"Decay time bins mismatch. Decay time bins: {decays.shape[1]}, time bins: {time_bins}.", None, None 

            if shift:
                # get sample decay curves from each kflow experiment, totoalling at 30 samples 
                # Use integer division to get number of samples per image, ensuring at least 1 sample
                samples_per_experiment = max(1, 30 // num_fovs)  # // operator performs integer division
                sample_decays, top_indices = _get_sample_decay_curves(decays, samples_per_experiment, 100000)
                # use the indices (essentially cell_number) to concatenate with image_name to form decay_id
                for i, index in enumerate(top_indices):
                    decay_curves[f"{fov_name}_{index}"] = sample_decays[i]

            else:
                # get all the decay curves for each cell
                # each cell is a row in the decays dataframe
                for index, row in decays.iterrows():
                    decay_curves[f"{fov_name}_{index}"] = row.values

    return error_msg, decay_curves, irf
       
       
