import errno
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    get_default_2D_decay_config,
    get_default_file_suffixes,
    get_default_laser_rate,
    get_fov_name_col,
    get_spc_output_suffix,
)
from src.decay_io import read_decay, read_decay_metadata
from src.emojis import happy_emoji, sad_emoji
from src.file_io import load_image


def load_data_suffix_widget(input_types, selected_channels, selected_ch_num_components, selected_feature_extractors):
    """Collect per-channel file suffixes and report missing or conflicting entries."""
    actual_file_suffix = {}
    error_lines = []
    t1_suffix_list = []
    mask_suffix_list = {}
    if any("prefitted" in input_type for input_type in input_types.values()):
        spc_output_suffix = get_spc_output_suffix()
    for i, (channel_key, channel_name) in enumerate(selected_channels.items()):
        input_type = input_types[channel_key]
        file_suffixes = get_default_file_suffixes(channel_key, input_type, selected_feature_extractors[channel_key])
        if len(file_suffixes) == 0:
            return "", f"No file suffixes found for {channel_name} {sad_emoji}"
        else:
            actual_file_suffix[channel_name] = file_suffixes

        st.subheader(f"File suffixes: {channel_name}")
        num_cols = 3
        cols = st.columns(num_cols)
        for j, (file_type, default_suffix) in enumerate(actual_file_suffix[channel_name].items()):
            col = cols[j % num_cols]
            with col:
                # only show the help message for the first file type of the first channel
                if i == 0 and j == 0:
                    help_msg = "The filenames are expected to have *exactly* two parts: *image_name + suffix*. All files from the same image should share the **same** image_name, with the only difference being the suffix."
                elif i == 0 and "prefitted" in input_type and file_type == "SPCImage t1":
                    help_msg = f"For other SPCImage output files (e.g. a1, t2), the suffixes are automatically generated based on the provided t1 suffix by replacing {spc_output_suffix['t1']} to get the others."
                else:
                    help_msg = None
                suffix = st.text_input(f"{file_type}", default_suffix, key=f"{channel_name}_{input_type}_{file_type}_suffix", help=help_msg)
                if suffix == "":
                    error_lines.append(f"Please provide a suffix for {file_type} in {channel_name}")
                else:
                    actual_file_suffix[channel_name][file_type] = suffix
            # Collect only entered suffixes for cross-channel checks; missing values
            # are already reported and must not count as duplicates.
            if suffix != "":
                if file_type == "SPCImage t1":
                    t1_suffix_list.append(suffix)
                elif file_type == "Mask":
                    mask_suffix_list[channel_name] = suffix
        if "prefitted" in input_type and not error_lines: # write the spc outputs' suffixes for this channel
            if channel_name in selected_ch_num_components and selected_ch_num_components[channel_name] != 0:
                num_components = selected_ch_num_components[channel_name]
                # t1 is already provided, so no need to generate the others
                if num_components == 1:
                    continue
                elif num_components == 2:
                    needed_suffix = ["a1", "t2"]
                elif num_components == 3:
                    needed_suffix = ["a1", "a2", "t2", "t3"]
                for key in needed_suffix:
                    actual_file_suffix[channel_name][key] = actual_file_suffix[channel_name]["SPCImage t1"].replace(spc_output_suffix["t1"], spc_output_suffix[key])

    # Check entered t1 suffixes for cross-channel duplicates.
    if len(set(t1_suffix_list)) != len(t1_suffix_list):
        error_lines.append(f"Duplicate t1 suffixes found: {t1_suffix_list}")

    # output info message for channels that share the same mask suffix
    mask_suffix_seen = {}
    for channel_name, mask_suffix in mask_suffix_list.items():
        if mask_suffix not in mask_suffix_seen:
            mask_suffix_seen[mask_suffix] = [channel_name]
        else:
            mask_suffix_seen[mask_suffix].append(channel_name)
    for mask_suffix, channel_names in mask_suffix_seen.items():
        if len(channel_names) > 1:
            st.info(f"Channels {channel_names} share the same mask suffix: {mask_suffix}")

    # Combine the collected problems into one de-duplicated, line-separated
    # message (blank line between items so st.error renders them on separate rows).
    error_msg = "\n\n".join(f"{line} {sad_emoji}" for line in dict.fromkeys(error_lines))

    # flatten the actual_file_suffix dictionary
    actual_file_suffix_dict = {}
    for channel_name, file_suffix_dict in actual_file_suffix.items():
        for file_type, file_suffix in file_suffix_dict.items():
            actual_file_suffix_dict[f"{channel_name}_{file_type}"] = file_suffix
    return actual_file_suffix_dict, error_msg

def _permission_denied_message(folder_path, err):
    """Explain a folder-listing failure, with platform-specific guidance when relevant.

    Permission and privacy restrictions can both block access. Add the macOS
    volume-access hint only for the matching platform and error/path signals.
    """
    msg = (
        f"⛔ **Permission denied** reading **{folder_path}**.\n\n"
        "Your account doesn't have permission to list this folder. If it's on a "
        "shared or network drive, check that you're connected and authorized to access it."
    )
    # On macOS, EPERM or a /Volumes path prompts guidance about volume access.
    if sys.platform == "darwin" and (
        getattr(err, "errno", None) == errno.EPERM or str(folder_path).startswith("/Volumes/")
    ):
        app = "FLIM Playground" if getattr(sys, "frozen", False) else "the terminal or IDE you launched it from"
        msg += (
            f"\n\n**On macOS**, {app} needs permission to read network or removable volumes. "
            "Enable it under System Settings → Privacy & Security → Full Disk Access, "
            "then fully quit and reopen it."
        )
    elif sys.platform == "win32":
        msg += (
            "\n\n**On Windows**, check the folder's permissions, or whether Controlled Folder "
            "Access (Windows Security → Ransomware protection) is blocking access to it."
        )
    return msg

@st.cache_data
def load_list_data_from_folder_widget(folder_path, file_suffix, num_cols=3):
    """Scan and display each FOV's files, keyed by image name.

    Per-FOV filenames must equal image_name + suffix. IRF and lifetime-standard
    files are shared across the dataset and matched by suffix alone.
    """

    valid_image_groups = {}

    # Single recursive scan to get all files

    path = Path(folder_path)

    # Probe folder access before rglob, which can suppress OS errors and make
    # an unreadable folder appear empty.
    if not path.exists():
        st.warning(f"Folder does not exist: **{folder_path}**. Please check the path.")
        return {}
    if not path.is_dir():
        st.warning(f"This path is not a folder: **{folder_path}**.")
        return {}
    try:
        os.listdir(folder_path)
    except PermissionError as e:
        st.error(f"{_permission_denied_message(folder_path, e)} {sad_emoji}")
        return {}
    except OSError as e:
        st.error(f"⛔ Could not read folder **{folder_path}**: {e} {sad_emoji}")
        return {}

    all_files = [str(file) for file in path.rglob("*") if file.is_file() and not file.name.startswith("fov_metadata") and not file.name.startswith("single_cell_features")]
    if len(all_files) == 0:
        st.warning(f"No files found in folder: **{folder_path}**.")
        return {}

    # Build lookup dictionaries for fast access
    files_by_name = {}  # exact filename -> list of file paths
    files_by_suffix = {}  # suffix -> list of file paths

    for file_path in all_files:
        filename = os.path.basename(file_path)

        # Index by exact filename
        if filename not in files_by_name:
            files_by_name[filename] = []
        files_by_name[filename].append(file_path)

        # Index by suffix for each suffix we care about
        for suffix in set(file_suffix.values()):
            if filename.endswith(suffix):
                if suffix not in files_by_suffix:
                    files_by_suffix[suffix] = []
                files_by_suffix[suffix].append(file_path)

    # use the first key to get the list of images (it does not matter which key to use, since they are all required, they should all be there)
    image_search_suffix = list(file_suffix.values())[0]
    image_files = files_by_suffix.get(image_search_suffix, [])
    if len(image_files) == 0:
        st.warning(f"No image files found with suffix: **{image_search_suffix}**.")
        return {}

    # Derive image names by removing the required file suffix.
    image_names = [os.path.basename(file).removesuffix(image_search_suffix) for file in image_files]
    # for each image name, build a widget card with the image name and the files that belong to it
    num_images = len(image_names)
    num_cols = min(num_cols, num_images)
    rows = (num_images + num_cols - 1) // num_cols

    if num_images > 0:
        st.markdown("##### :green[Fields of view:] \n")

    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx >= num_images:
                break
            image_name = image_names[img_idx]
            image_group = {}
            missing_keys = []
            duplicate_keys = []
            # get the list of files that belong to this image
            for key, suffix in file_suffix.items():
                # Exact match for per-FOV files; suffix-only for IRF and Fluorescence Lifetime Standard (global per dataset)
                if "IRF" not in key and "Fluorescence Lifetime Standard" not in key:
                    filename = image_name + suffix
                    matched_files = files_by_name.get(filename, [])
                else:
                    matched_files = files_by_suffix.get(suffix, [])

                if len(matched_files) != 1:
                    if len(matched_files) > 1:
                        duplicate_keys.append(key) # more than one file found
                    else:
                        missing_keys.append(key) # no matching file found
                else:
                    image_group[key] = matched_files[0]


            # create the card
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{image_name}**")
                    if missing_keys or duplicate_keys:
                        st.write("❌ Missing or duplicate files:")
                        for key in missing_keys:
                            if "IRF" not in key and "Fluorescence Lifetime Standard" not in key:
                                st.write(f"- Missing {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Missing {key} with suffix: {file_suffix[key]}")
                        for key in duplicate_keys:
                            if "IRF" not in key and "Fluorescence Lifetime Standard" not in key:
                                st.write(f"- Duplicate {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Duplicate {key} with suffix: {file_suffix[key]}")

                    else:
                        st.write("✅ All files found.")


            if missing_keys == [] and duplicate_keys == []:
                valid_image_groups[image_name] = image_group

    return valid_image_groups

def preview_metadata_widget(metadata_df, num_cols=3):
    """
    Display the feature groups to be extracted.
    """
    # if there are more than num_cols rows, write the first num_cols rows, else write all rows
    if len(metadata_df) > num_cols:
        st.write(metadata_df.head(num_cols))
    else:
        st.write(metadata_df)

def export_metadata_widget(metadata_df, folder_path):
    # Export one metadata row per FOV to the selected folder.
    confirm_export = st.button("Export FOV Metadata as CSV", help=f"Export the fov metadata as one csv file (one fov per row) to {folder_path}", key="export_metadata_button")
    if confirm_export:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = os.path.join(folder_path, f"fov_metadata_{time_stamp}.csv")
        try:
            metadata_df.to_csv(csv_file_path) # Save the DataFrame
        except Exception as e:
            st.error(f"Error exporting the fov metadata: {e}. Is the previous metadata file open in another program? {sad_emoji}")
            return
        st.success(f"FOV metadata exported successfully to {csv_file_path} {happy_emoji}")
        st.session_state["last_extracted_metadata"] = metadata_df
        st.session_state["last_extracted_metadata_filepath"] = csv_file_path

# Cache validation by the columns each check reads. Channel assignments
# mutate fov_df between calls and must not invalidate other channels' scans.


def _column_values(fov_df, column_name):
    """Return a column as a hashable tuple, preserving missing values.

    ``tolist`` converts numpy scalars for faster cache hashing. Keep NaNs as
    missing paths so readers can report them without attempting a "nan" file.
    """
    return tuple(fov_df[column_name].tolist())


def _fov_labels(fov_df):
    """Return FOV labels as strings for errors and fast cache hashing.

    Stringifying also avoids generic hashing of numpy integer index values.
    """
    fov_name_col = get_fov_name_col()
    source = fov_df[fov_name_col] if fov_name_col in fov_df.columns else fov_df.index
    return tuple(str(label) for label in source)


def _files_by_group_msg(header, label, mapping):
    """Spell out which files disagreed: one group per distinct value, with the
    offending basenames listed two per line underneath."""
    msg = header
    for value, files in mapping.items():
        msg += f"- {label} {value} ({len(files)} file(s)):\n"
        basenames = [os.path.basename(f) for f in files]
        for i in range(0, len(basenames), 2):
            msg += f"  - {', '.join(basenames[i : i + 2])}\n"
    return msg


class _DecayScan(NamedTuple):
    """What reading the decay files tells us, before any channel-specific
    interpretation. Built by _scan_decay_files, read by check_raw_decay_data."""

    shape_list: list
    shape_to_files: dict
    laser_rep_time_list: list
    laser_rep_time_to_files: dict
    empty_fov_labels: list
    channel_has_signal: object  # (C,) bool array, or None if no 4D FOV was read
    preview_images: object  # (C, Y, X) float64 photon counts of the first 4D FOV, or None


@st.cache_data(show_spinner="Reading decay files...")
def _scan_decay_files(decay_paths, fov_labels):
    """Read decay files and return ``(error_msg, _DecayScan | None)``.

    Cache by file paths and FOV labels, without channel-specific state, so
    channels sharing multi-detector files reuse one decode. Channel assignment and channel-specific
    messages belong in ``check_raw_decay_data``.
    """
    shape_list = []
    shape_to_files = {}  # shape -> list of decay file paths
    laser_rep_time_list = []
    laser_rep_time_to_files = {}  # laser_rep_time -> list of decay file paths
    empty_fov_labels = []
    channel_has_signal = None  # set on first 4D decay; used if all FOVs agree on 4D shape
    preview_images = None  # (C, Y, X) intensity of the first 4D FOV, for the channel preview
    for decay_path, fov_label in zip(decay_paths, fov_labels):
        error_msg, decay_data = read_decay(decay_path)
        if error_msg != "":
            return error_msg, None
        shape = decay_data.shape
        shape_list.append(shape)
        shape_to_files.setdefault(shape, []).append(decay_path)
        error_msg, laser_rep_time = read_decay_metadata(decay_path)
        if error_msg != "":
            return error_msg, None
        laser_rep_time_list.append(laser_rep_time)
        laser_rep_time_to_files.setdefault(laser_rep_time, []).append(decay_path)

        # Only aggregate when this row matches the first row's shape (avoids
        # resizing channel_has_signal if later files differ in channel count).
        if len(shape) == 4 and shape == shape_list[0]:
            if channel_has_signal is None:
                channel_has_signal = np.zeros(shape[0], dtype=bool)
                # Sum the first 4D FOV while it is in memory to avoid a preview reread.
                # Use float64 for photon totals.
                preview_images = np.sum(decay_data, axis=-1, dtype=np.float64)
            if not np.any(decay_data):
                empty_fov_labels.append(fov_label)
            else:
                for c in range(shape[0]):
                    if np.any(decay_data[c]):
                        channel_has_signal[c] = True

    return "", _DecayScan(
        shape_list=shape_list,
        shape_to_files=shape_to_files,
        laser_rep_time_list=laser_rep_time_list,
        laser_rep_time_to_files=laser_rep_time_to_files,
        empty_fov_labels=empty_fov_labels,
        channel_has_signal=channel_has_signal,
        preview_images=preview_images,
    )


def check_raw_decay_data(fov_df, channel_name):
    """Interpret cached decay data for one channel.

    Return ``(error_msg, available_channels, shape, laser_rep_time, preview_images)``.
    On success, ``shape`` is (Y, X, T) for both 3D and 4D inputs. Preview images
    are (C, Y, X) photon totals for the first FOV, or None for 3D inputs/errors.

    Keep this wrapper uncached so assignments reuse ``_scan_decay_files`` without
    rereading files for any channel.
    """
    decay_column_name = f"{channel_name}_Decay"
    mask_column_name = f"{channel_name}_Mask"
    if decay_column_name not in fov_df.columns:
        return "Error: No decay data found. Please check the data.", [], None, None, None
    # Only check mask-column presence here; mask paths are not scan-cache inputs.
    if mask_column_name not in fov_df.columns:
        return "Error: No mask data found. Please check the data.", [], None, None, None

    error_msg, scan = _scan_decay_files(
        _column_values(fov_df, decay_column_name), _fov_labels(fov_df)
    )
    if error_msg != "":
        return error_msg, [], None, None, None
    if len(scan.shape_list) == 0:
        return f"Error: No {channel_name} decay files to check.", [], None, None, None

    # check for the consistency of the shape, a tuple
    if len(set(scan.shape_list)) > 1:
        return _files_by_group_msg(
            f"Inconsistent decay data shapes found for {channel_name} decay: \n",
            "Shape",
            scan.shape_to_files,
        ), [], None, None, None
    if len(set(scan.laser_rep_time_list)) > 1:
        return _files_by_group_msg(
            f"Inconsistent laser rep time found for {channel_name} decay: \n",
            "Laser rep time",
            scan.laser_rep_time_to_files,
        ), [], None, None, None

    # get the first shape: CYXT or YXT
    shape = scan.shape_list[0]
    laser_rep_time = scan.laser_rep_time_list[0]
    if len(shape) == 3:
        # single channel per file: nothing to assign, so nothing to preview
        return "", [-1], shape, laser_rep_time, None
    elif len(shape) == 4:
        if len(scan.empty_fov_labels) > 0:
            listed = ", ".join(scan.empty_fov_labels)
            return (
                f"{len(scan.empty_fov_labels)} field(s) of view have entirely zero "
                f"{channel_name} decay data ({listed}). Please check the data."
            ), [], None, None, None
        # Each FOV has signal, so at least one channel is available for assignment.
        non_zero_channels = [c for c in range(shape[0]) if scan.channel_has_signal[c]]
        return "", non_zero_channels, shape[1:], laser_rep_time, scan.preview_images
    # Preserve the error return structure for unexpected dimensions.
    return (
        f"Error: Unexpected {channel_name} decay data shape {shape}. "
        "Expected 3 or 4 dimensions."
    ), [], None, None, None


@st.cache_data(show_spinner="Reading 2D decay files...")
def _scan_2d_decay_files(decay_paths):
    """Return ``(error_detail, time_bins)`` from the first 2D decay CSV.

    Only the first file is read, though the full path tuple identifies the cached
    result. The caller adds channel context to errors.
    """
    for decay_path in decay_paths:
        try:
            decay_data = pd.read_csv(decay_path, header=None)
        except Exception as e:
            return str(e), None
        return "", decay_data.shape[1]
    return "no decay files to check", None


def check_raw_2D_decay_data(fov_df, channel_name):
    decay_column_name = f"{channel_name}_Decay"
    if decay_column_name not in fov_df.columns:
        return f"Error: Decay data path not found for {channel_name}", None
    error_detail, time_bins = _scan_2d_decay_files(_column_values(fov_df, decay_column_name))
    if error_detail != "":
        return f"Error reading decay data for {channel_name}: {error_detail}", None
    return "", time_bins


@st.cache_data(show_spinner="Reading intensity images...")
def _scan_intensity_images(intensity_paths, mask_paths):
    """Check that each intensity image and mask are 2D and have equal shapes.

    Cache by both path tuples because both files are read. Return
    ``(error_msg, dimension_list)`` with file-specific errors.
    """
    dimension_list = []
    for image_path, mask_path in zip(intensity_paths, mask_paths):
        try:
            image_data = load_image(image_path)
        except Exception as e:
            return f"Error reading intensity image: {image_path}: {e}", None
        if len(image_data.shape) != 2:
            return f"Error: Intensity image {image_path} is not a 2D array", None

        dimension_list.append(image_data.shape)
        # check for the consistency of the shape between the intensity image and the mask image
        try:
            mask_data = load_image(mask_path)
        except Exception as e:
            return f"Error reading mask image: {mask_path}: {e}", None
        if len(mask_data.shape) != 2:
            return f"Error: Mask image {mask_path} is not a 2D array", None
        if image_data.shape != mask_data.shape:
            return f"Error: Intensity image {image_path} and mask image {mask_path} have different shapes: {image_data.shape} != {mask_data.shape}", None
    return "", dimension_list


def check_raw_intensity_data(fov_df, channel_name):
    intensity_column_name = f"{channel_name}_Intensity (2D)"
    mask_column_name = f"{channel_name}_Mask"
    if intensity_column_name not in fov_df.columns:
        return f"Error: Intensity image path not found for {channel_name}", None
    if mask_column_name not in fov_df.columns:
        return f"Error: Mask path not found for {channel_name}", None

    error_msg, dimension_list = _scan_intensity_images(
        _column_values(fov_df, intensity_column_name),
        _column_values(fov_df, mask_column_name),
    )
    if error_msg != "":
        return error_msg, None

    if len(set(dimension_list)) > 1:
        return f"Inconsistent fov dimensions found for channel {channel_name}. Please check the data.", None
    elif len(dimension_list) == 0:
        return f"No fov dimensions found for channel {channel_name}. Please check the data.", None
    else:
        return "", dimension_list[0]

def clear_folder_scan_caches():
    """Clear folder listing and all raw-data caches for the Rescan folder action.

    Clear the cached readers; their check_raw_* wrappers are uncached.
    """
    load_list_data_from_folder_widget.clear()
    _scan_decay_files.clear()
    _scan_2d_decay_files.clear()
    _scan_intensity_images.clear()

def _inconsistent_selected(x):
    return f"Inconsistent {x} found for the selected channels. Please check the data."


def _none_selected(x):
    return f"No {x} found for the selected channels. Please check the data."


def _to_display_image(intensity_image, high_percentile=99.5):
    """Photon counts -> 8-bit grayscale for st.image, stretched from 0 to a percentile.

    Clipping at a high percentile rather than the max stops a handful of hot pixels from
    crushing the rest of the image to black. The floor stays at 0, not the minimum,
    because zero photons genuinely means no signal.
    """
    vmax = np.percentile(intensity_image, high_percentile)
    if vmax <= 0:  # a channel whose signal is all in the tail, or nothing at all
        vmax = intensity_image.max()
    if vmax <= 0:
        return np.zeros(intensity_image.shape, dtype=np.uint8)
    return (np.clip(intensity_image / vmax, 0, 1) * 255).astype(np.uint8)


def _render_channel_preview(fov_df, preview_images, available_channels, selected_channel):
    """Thumbnails of every non-zero channel in the first FOV, so the channel assigned
    above can be checked against what the data actually looks like.

    Each channel is stretched independently so a dim one stays legible; the photon total
    in each caption carries the relative-brightness cue that independent stretching drops.
    """
    if preview_images is None:
        return  # 3D decay, or a read error check_raw_decay_data has already reported
    fov_name_col = get_fov_name_col()
    fov_label = fov_df.iloc[0][fov_name_col] if fov_name_col in fov_df.columns else fov_df.index[0]
    st.caption(f"First FOV: {fov_label}")
    preview_cols = st.columns(len(available_channels))
    for j, channel_no in enumerate(available_channels):
        if channel_no >= preview_images.shape[0]:
            continue
        intensity_image = preview_images[channel_no]
        with preview_cols[j]:
            # A channel can be all-zero in this FOV yet have signal in a later one, since
            # channel_has_signal ORs across FOVs: that shows up as black, "0 photons".
            marker = " \u2705" if channel_no == selected_channel else ""
            st.image(
                _to_display_image(intensity_image),
                width="stretch",
                caption=f"ch {channel_no + 1}{marker} \u2014 {intensity_image.sum():,.0f} photons",
            )


def check_assign_channel_widget(fov_df, selected_channels, flim_decay_input_type, imaging_modalities, selected_ch_feature_extractors, duration=None, time_bins=None):
    error_msg = ""
    time_bins_list = []
    laser_rep_time_list = []
    fov_dimensions_list = []
    num_cols = len(selected_channels)
    cols = st.columns(num_cols)
    has_flim = has_3_4D_decay = has_intensity_only = False
    for i, (channel_key, channel_name) in enumerate(selected_channels.items()):
        imaging_modality = imaging_modalities[channel_key]
        if imaging_modality == "Intensity-only":
            has_intensity_only = True
            error_msg, fov_dimensions = check_raw_intensity_data(fov_df, channel_name)
            if error_msg == "":
                fov_dimensions_list.append(fov_dimensions)
            else:
                return error_msg, None
        elif imaging_modality == "FLIM":
            if "prefitted" in flim_decay_input_type:
                feature_extractors = selected_ch_feature_extractors[channel_key]
                if "Lifetime fit" in feature_extractors and len(feature_extractors) == 1:
                    has_flim = False
                else:
                    has_flim = True
            else:
                has_flim = True

            if has_flim:
                decay_col_name = f"{channel_name}_Decay"
                if decay_col_name not in fov_df.columns:
                    return "Error: File paths for decay data are not provided.", None
                if flim_decay_input_type == "Decay (2D)":
                    if duration is not None:
                        fov_df["duration"] = duration
                    else:
                        return "Error: Duration is not provided.", None
                    if time_bins is not None:
                        error_msg, time_bins = check_raw_2D_decay_data(fov_df, channel_name)
                        if error_msg == "":
                            time_bins_list.append(time_bins)
                        else:
                            return error_msg, None
                else: # 3/4D decay
                    has_3_4D_decay = True
                    with cols[i]:
                        error_msg, available_channels, shape, laser_rep_time, preview_images = check_raw_decay_data(fov_df, channel_name)
                        if error_msg == "":
                            if len(available_channels) == 1:
                                fov_df[f"{channel_name}_channel"] = available_channels[0]
                            else:
                                human_readable_channel_nos = [channel_no + 1 for channel_no in available_channels]
                                human_readable_channel_no = st.selectbox(f"Select the channel for {channel_name} decay", human_readable_channel_nos, key=f"{channel_name}_channel_selectbox")
                                fov_df[f"{channel_name}_channel"] = human_readable_channel_no - 1
                                _render_channel_preview(fov_df, preview_images, available_channels, human_readable_channel_no - 1)
                            time_bins_list.append(shape[-1])
                            fov_dimensions_list.append(shape[:-1])
                            laser_rep_time_list.append(laser_rep_time)
                        else:
                            return error_msg, None
        else:
            continue

    if len(set(time_bins_list)) > 1:
        return _inconsistent_selected("time bins"), None
    elif len(time_bins_list) == 0:
        if has_flim:
            return _none_selected("time bins"), None
    else:
        fov_df["time_bins"] = time_bins_list[0]

    if len(set(laser_rep_time_list)) > 1:
        return _inconsistent_selected("laser rep time"), None
    elif len(laser_rep_time_list) == 0:
         # in 2d decay, the laser rep time is given by the user, so no way to check it here
        if has_3_4D_decay:
            return _none_selected("laser rep time"), None
    else:
        fov_df["duration"] = laser_rep_time_list[0]

    if len(set(fov_dimensions_list)) > 1:
        return _inconsistent_selected("fov spatial dimensions"), None
    elif len(fov_dimensions_list) == 0:
        if has_intensity_only or has_3_4D_decay:
            return _none_selected("fov dimensions"), None
    else:
        # Store as string to avoid hashing issues in caching
        fov_df["fov_dimensions"] = [str(fov_dimensions_list[0])] * len(fov_df)

    return error_msg, fov_df

def lifetime_data_config_widget(selected_feature_extractors, input_type):
    fit_free = False
    duration = time_bins = laser_rate = None
    for _, extractors in selected_feature_extractors.items():
        if "Lifetime fit free" in extractors:
            fit_free = True
    if input_type == "Decay (2D)":
        default_2D_decay_duration, default_2D_decay_time_bins = get_default_2D_decay_config()
        cols = st.columns(3 if fit_free else 2)
        with cols[0]:
            duration = st.number_input("Duration (**ns**)", value=default_2D_decay_duration, min_value=0.0, max_value=100.0, key="2D_decay_duration")
        with cols[1]:
            time_bins = st.number_input("Time bins", value=default_2D_decay_time_bins, min_value=10, key="2D_decay_time_bins")
        if fit_free:
            default_laser_rate = get_default_laser_rate(input_type)
            with cols[2]:
                laser_rate = st.number_input("Laser rate **(GHz)**", value=default_laser_rate, min_value=0.0, max_value=1.0, key="2D_decay_laser_rate")
    else:
        if fit_free:
            default_laser_rate = get_default_laser_rate(input_type)
            laser_rate = st.number_input("Laser rate **(GHz)**", value=default_laser_rate, min_value=0.0, max_value=1.0, key="laser_rate")
    return duration, time_bins, laser_rate