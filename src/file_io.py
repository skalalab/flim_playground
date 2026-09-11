import codecs
import numpy as np
import pathlib
from pathlib import Path
import tifffile
from typing import Union
from src.decay_io import read_decay, read_ptu_reference
from src.config import get_fov_name_col
import pandas as pd
import os

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
    raise ValueError(f"Unsupported file extension '{path.suffix}'. Supported: .asc, .tiff, .tif")

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
    # Select the brightest n_samples curves below max_intensity.
    decay_intensity = np.sum(decays, axis=1)
    sorted_indices = np.argsort(decay_intensity)[::-1]
    filtered_indices = [idx for idx in sorted_indices if decay_intensity[idx] < max_intensity]

    # Fallback: if intensity filter removed all cells, ignore the cap and
    # take the n_samples brightest curves so shift estimation still works.
    if len(filtered_indices) == 0:
        filtered_indices = list(sorted_indices)

    if len(filtered_indices) < n_samples:
        top_indices = filtered_indices
    else:
        top_indices = filtered_indices[:n_samples]
    return decays.iloc[top_indices].values, top_indices

def _decay_shape_msg(expected, shape):
    return f"Decay data mismatch. Expected {expected}, got {shape}."


def _time_bins_msg(actual, expected):
    return f"Decay time bins mismatch. Decay time bins: {actual}, time bins: {expected}."


def get_decay_curves(metadata_df, input_type, channel_name, time_bins, shift=True):

    error_msg = ""
    decay_curves = {}
    fov_name_col = get_fov_name_col()
    
    # Handle iteration for both DataFrame and Series cases
    if isinstance(metadata_df, pd.Series):
        # Process single row (Series)
        rows_to_process = [(0, metadata_df)]
        num_fovs = 1
    else:
        # Process DataFrame rows
        rows_to_process = list(metadata_df.iterrows())
        num_fovs = len(metadata_df)
    
    for i, row in rows_to_process:
        fov_name = row[fov_name_col]
        decay_path = row.get(f'{channel_name}_Decay', None)
        if "Decay (3/4D)" in input_type:
            mask_path = row.get(f'{channel_name}_Mask', None)
            try:
                mask = load_image(mask_path)
            except Exception as e:
                return f"Error reading the mask file for {fov_name_col} {fov_name} at {mask_path}: {e}", None  
            channel_no = row.get(f'{channel_name}_channel', None)
            if channel_no is None:
                return f"Error: Channel number not found for {fov_name_col} {fov_name}", None      
            try:
                error_msg, decay = read_decay(decay_path, channel_no)
                if error_msg != "":
                    return error_msg, None
            except Exception as e:
                return f"Error reading the decay file for {fov_name_col} {fov_name} at {decay_path}: {e}", None
            if len(decay.shape) != 3:
                return _decay_shape_msg("3D data (XYT)", decay.shape), None
            if decay.shape[-1] != time_bins:
                return _time_bins_msg(decay.shape[2], time_bins), None
            if len(mask.shape) != 2:
                return f"Mask data mismatch. Expected 2D data, got {mask.shape}.", None
            if decay.shape[0] != mask.shape[0] or decay.shape[1] != mask.shape[1]:
                return f"Dimension mismatch: Decay data {decay.shape[:2]} vs mask {mask.shape}", None
            
            if shift:
                # binarize the mask
                binary_mask = np.where(mask > 0, 1, 0)
                # Sum masked pixels into one decay curve per FOV, retaining time bins.
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
                decays = pd.read_csv(decay_path, header=None)    
            except Exception as e:
                return f"Error reading the histogram file for {fov_name_col} {fov_name} at {decay_path}: {e}", None
            non_numeric = decays.select_dtypes(exclude="number").columns.tolist()
            if non_numeric:
                return f"Decay file {os.path.basename(decay_path)} contains non-numeric columns. Expected all columns to be numeric time bins.", None

            if len(decays.shape) != 2:
                return _decay_shape_msg("2D data", decays.shape), None
                
            if decays.shape[1] != time_bins:
                return _time_bins_msg(decays.shape[1], time_bins), None 
            
            if shift:
                # Divide a target of 30 samples across FOVs, allowing at least one each.
                samples_per_experiment = max(1, 30 // num_fovs)
                sample_decays, top_indices = _get_sample_decay_curves(decays, samples_per_experiment, 100000)
                # Combine the FOV name and original row index into each cell ID.
                for i, index in enumerate(top_indices):
                    decay_curves[f"{fov_name}_{index}"] = sample_decays[i]

            else:
                # get all the decay curves for each cell
                # each cell is a row in the decays dataframe
                for index, row in decays.iterrows():
                    decay_curves[f"{fov_name}_{index}"] = row.values

    return error_msg, decay_curves

def _validate_reference_timing(reference, metadata_df, time_bins, label):
    """Require the PTU acquisition grid; never resample a calibration curve."""
    rows = [metadata_df] if isinstance(metadata_df, pd.Series) else (
        row for _, row in metadata_df.iterrows())
    ref_width = reference.duration / reference.time_bins
    for row in rows:
        mismatches = []
        sample_bins = row.get("time_bins", time_bins)
        if reference.time_bins != time_bins or reference.time_bins != sample_bins:
            mismatches.append(f"time bins mismatch: reference={reference.time_bins}, "
                              f"sample={sample_bins} (requested={time_bins})")
        try:
            duration = float(row.get("duration", np.nan))
            sample_width = duration / float(sample_bins)
            if not np.isfinite(sample_width) or sample_width <= 0:
                raise ValueError
        except (TypeError, ValueError, ZeroDivisionError):
            mismatches.append("sample duration and time bins must be positive finite numbers to validate the PTU bin width")
        else:
            if not np.isclose(ref_width, sample_width, rtol=1e-5, atol=0):
                mismatches.append(f"bin width mismatch: reference={ref_width:.12g} ns, "
                                  f"sample={sample_width:.12g} ns")
        laser_rate = row.get("laser_rate")
        if laser_rate is not None and not pd.isna(laser_rate):
            try:
                laser_rate = float(laser_rate)
            except (ValueError, TypeError):
                mismatches.append(f"invalid sample laser frequency: {laser_rate}")
            else:
                if not np.isclose(reference.laser_rate, laser_rate, rtol=1e-5, atol=0):
                    mismatches.append(f"laser frequency mismatch: reference={reference.laser_rate * 1000:.12g} MHz, "
                                      f"sample={laser_rate * 1000:.12g} MHz")
        if mismatches:
            return f"Error: {label}: " + "; ".join(mismatches) + "."
    return ""


def get_lifetime_standard(metadata_df, channel_name, time_bins):
    """Return ``(error_msg, (reference_image, time_axis))`` for calibration.

    PTU curves use shape (1, 1, time_bins). TIFF/ASC images retain their original
    layout; use the CSV's time axis when present or infer one unambiguous axis.
    """
    row = metadata_df if isinstance(metadata_df, pd.Series) else metadata_df.iloc[0]
    path = row.get(f"{channel_name}_Fluorescence Lifetime Standard")
    label = f"Fluorescence lifetime standard for {channel_name}"
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        return f"Error: {label} file path not specified.", None
    if Path(path).suffix.lower() == ".ptu":
        err, ref = read_ptu_reference(path)
        if err:
            return err, None
        err = _validate_reference_timing(ref, metadata_df, time_bins, label)
        if err:
            return err, None
        return "", (ref.curve.reshape(1, 1, -1), 2)
    try:
        data = load_image(path)
        if data.ndim != 3:
            return f"{label} must be 3D, got {data.ndim} with shape {data.shape}.", None
        axis = row.get(f"{channel_name}_fluorescence_lifetime_standard_time_axis")
        if axis is not None and not pd.isna(axis):
            if int(axis) != float(axis) or int(axis) not in range(3) or data.shape[int(axis)] != time_bins:
                return f"Invalid time axis {axis} for {label}: shape={data.shape}, time bins={time_bins}.", None
            axis = int(axis)
        else:
            matches = data.shape.count(time_bins)
            if matches == 0:
                return f"Cannot find the time axis ({time_bins} bins) for {label} dimensions: {data.shape}.", None
            if matches > 1:
                return f"Ambiguous time axis for {label} dimensions: {data.shape}.", None
            axis = data.shape.index(time_bins)
        return "", (data, axis)
    except Exception as e:
        return f"Error reading {label} at {path}: {e}", None


def get_irf(metadata_df, channel_name, time_bins):
    # Handle both DataFrame and Series cases
    if isinstance(metadata_df, pd.Series):
        # metadata_df is already a single row (Series)
        first_row = metadata_df
    else:
        # metadata_df is a DataFrame, get the first row
        first_row = metadata_df.iloc[0]
      
    irf_path = first_row.get(f'{channel_name}_IRF', None)
    
    if not isinstance(irf_path, (str, os.PathLike)) or not str(irf_path).strip():
        return f"Error: IRF file path not specified for {channel_name}.", None

    if Path(irf_path).suffix.lower() == ".ptu":
        error_msg, reference = read_ptu_reference(irf_path)
        if error_msg:
            return error_msg, None
        error_msg = _validate_reference_timing(reference, metadata_df, time_bins, f"IRF for {channel_name}")
        if error_msg:
            return error_msg, None
        return "", reference.curve
    
    try:
        if str(irf_path).endswith(".csv"):
            irf = np.loadtxt(irf_path, delimiter=",").flatten()
        else:
            irf = np.loadtxt(irf_path)
    except Exception as e:
        return f"Error: IRF file not found or unreadable for {channel_name} at {irf_path}: {e}", None
    if irf.ndim != 1:
        return f"IRF must be 1D. Got {irf.ndim}D array with shape {irf.shape}.", None
    if len(irf) != time_bins:
        return f"IRF length mismatch with specified time bins. IRF length: {len(irf)}, time bins: {time_bins}.", None
    return "", irf
