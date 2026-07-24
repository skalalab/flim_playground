import json
import os
import time
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.config import (
    get_channel_names,
    get_current_profile_name,
    get_decay_input_type,
    get_derived_features,
    get_fit_free_calibration_method,
    get_fixed_lifetimes,
    get_fov_name_col,
    get_imaging_modality,
    get_input_types,
    get_num_components,
    get_selected_feature_extractors,
)
from src.config_watch import notify_on_config_change
from src.emojis import happy_emoji, sad_emoji
from src.file_io import load_image
from src.metadata import parse_metadata_file
from src.navigation import render_top_menu
from src.widgets.category_widgets import (
    check_and_merge_df_widget,
    find_available_dfs_widget,
    map_categories_to_labels_widget,
)
from src.widgets.lifetime_widgets import choose_shift_widget, fit_options_widget
from src.widgets.metadata_widgets import (
    check_assign_channel_widget,
    clear_folder_scan_caches,
    export_metadata_widget,
    lifetime_data_config_widget,
    load_data_suffix_widget,
    load_list_data_from_folder_widget,
    preview_metadata_widget,
)
from src.widgets.numeric_extraction_widgets import fov_extraction_widget

# --- Step identity ---------------------------------------------------------
# Single source of truth for the three workflow steps, used for both the radio
# and the dispatch below. Selecting a step returns exactly one of these strings.
STEP_FOV = "FOV Metadata Extraction"
STEP_NUMERIC = "Numeric Feature Extraction (fitting, phasor, etc.)"
STEP_CATEGORICAL = "Categorical Feature Extraction (e.g. treatment)"
STEPS = [STEP_FOV, STEP_NUMERIC, STEP_CATEGORICAL]


# --- Cross-step context ----------------------------------------------------
@dataclass
class ExtractionContext:
    """Config-derived values that are the same across all three steps, resolved
    once from the active profile and passed explicitly to the step renderers and
    helpers (instead of being read from module globals)."""
    channel_names: dict
    input_types: dict
    imaging_modalities: dict
    has_flim: bool
    decay_input_type: str
    ch_num_components: dict
    selected_ch_feature_extractors: dict
    fov_name_col: str
    fit_free_calibration_method: object
    fluorescence_lifetime_standard_lifetime: object


def build_context():
    """Resolve all config-derived, step-independent values from the active profile."""
    channel_names = get_channel_names()
    input_types = get_input_types(channel_names.keys())
    imaging_modalities = get_imaging_modality(channel_names.keys())
    has_flim = "FLIM" in imaging_modalities.values()
    decay_input_type = get_decay_input_type()
    ch_num_components = get_num_components(input_types, channel_names.keys())
    selected_ch_feature_extractors = get_selected_feature_extractors(input_types, channel_names.keys())
    fov_name_col = get_fov_name_col()
    # get_fit_free_calibration_method returns a 2-tuple (method, standard_lifetime)
    fit_free_calibration_method, fluorescence_lifetime_standard_lifetime = get_fit_free_calibration_method(decay_input_type)
    return ExtractionContext(
        channel_names=channel_names,
        input_types=input_types,
        imaging_modalities=imaging_modalities,
        has_flim=has_flim,
        decay_input_type=decay_input_type,
        ch_num_components=ch_num_components,
        selected_ch_feature_extractors=selected_ch_feature_extractors,
        fov_name_col=fov_name_col,
        fit_free_calibration_method=fit_free_calibration_method,
        fluorescence_lifetime_standard_lifetime=fluorescence_lifetime_standard_lifetime,
    )


def init_session_state():
    if "last_extracted_metadata" not in st.session_state:
        st.session_state["last_extracted_metadata"] = None
    if "last_extracted_metadata_filepath" not in st.session_state:
        st.session_state["last_extracted_metadata_filepath"] = None
    if "choosing_shift" not in st.session_state:
        st.session_state["choosing_shift"] = False
    if "shift_ready" not in st.session_state:
        st.session_state["shift_ready"] = False


# --- FOV Metadata Extraction helpers ---------------------------------------
def validate_folder_path(folder_path):
    """Validate folder path and return appropriate error message"""
    if folder_path == "":
        st.info("Please provide a folder path.")
        return False
    if not os.path.isdir(folder_path):
        st.error(f"Folder not found! Please check the path. {sad_emoji}")
        return False
    return True


def load_and_validate_fovs(folder_path, actual_file_suffix):
    """Load FOVs from folder and validate"""
    fovs = load_list_data_from_folder_widget(folder_path, file_suffix=actual_file_suffix)
    if len(fovs) == 0:
        st.warning("No data found in the folder. Please check the path and the file suffixes.")
        return None

    st.success(f"Fields of View with ✅ are loaded successfully {happy_emoji}. FOVs with ❌ (if any) will **not** be recorded. Here is the preview of the FOVs and metadata recorded:")
    return fovs


def prepare_fov_dataframe(fovs, selected_channels, selected_ch_num_components, ctx):
    """Prepare FOV dataframe with channel information"""
    fov_df = pd.DataFrame.from_dict(fovs, orient="index")

    # Set index name and reset to column
    fov_df.index.name = ctx.fov_name_col
    fov_df.reset_index(inplace=True)

    # Add channel information
    for channel_key, channel_name in selected_channels.items():
        fov_df[f"{channel_name}_input_type"] = ctx.input_types[channel_key]
        fov_df[f"{channel_name}_imaging_modality"] = ctx.imaging_modalities[channel_key]
        for feature_extractor in ctx.selected_ch_feature_extractors[channel_key]:
            fov_df[f"{channel_name}_{feature_extractor}"] = True
        if ctx.has_flim and channel_name in selected_ch_num_components:
            fov_df[f"{channel_name}_num_components"] = selected_ch_num_components[channel_name]
        # Write fixed-lifetime columns from config defaults (Step 1)
        if "Lifetime fit" in ctx.selected_ch_feature_extractors.get(channel_key, []):
            fixed_lts = get_fixed_lifetimes(channel_key, ctx.input_types[channel_key])
            for t_key in ["t1", "t2", "t3"]:
                val = fixed_lts.get(t_key)  # None or float
                fov_df[f"{channel_name}_fixed_{t_key}"] = val

    # Serialize derived-feature definitions into one JSON column, repeated per row
    # (a global setting, like laser_rate). This bakes the formulas into the
    # metadata CSV so re-running a saved CSV reproduces the same "Derived: *"
    # columns regardless of later config edits (see src/derived_features.py).
    fov_df["derived_features"] = json.dumps(get_derived_features())

    return fov_df


def validate_fluorescence_lifetime_standard_per_channel(fov_df, selected_channels, fit_free_calibration_method, time_bins, fluorescence_lifetime_standard_lifetime):
    """Validate and add per-channel fluorescence lifetime standard info if needed"""
    fov_df["fit_free_calibration_method"] = fit_free_calibration_method
    if fit_free_calibration_method != "Fluorescence Lifetime Standard":
        return "", fov_df

    # lifetime is shared across channels
    fov_df["fluorescence_lifetime_standard_lifetime"] = fluorescence_lifetime_standard_lifetime

    for channel_name in selected_channels.values():
        ref_col = f"{channel_name}_Fluorescence Lifetime Standard"
        if ref_col not in fov_df.columns:
            # Channel may not use fit free; skip
            continue
        unique_paths = fov_df[ref_col].dropna().unique().tolist()
        if len(unique_paths) != 1:
            return f"Fluorescence lifetime standard file path column {ref_col} is not consistent across FOVs.", fov_df
        fluorescence_lifetime_standard_file_path = unique_paths[0]
        # Check dimensions of fluorescence lifetime standard file
        try:
            fluorescence_lifetime_standard_data = load_image(fluorescence_lifetime_standard_file_path)
            fluorescence_lifetime_standard_shape = fluorescence_lifetime_standard_data.shape
            if len(fluorescence_lifetime_standard_shape) != 3:
                return f"Fluorescence lifetime standard file for {channel_name} must be 3D, got {len(fluorescence_lifetime_standard_shape)} with shape {fluorescence_lifetime_standard_shape}", fov_df
            matched_time_bins = fluorescence_lifetime_standard_shape.count(time_bins)
            if matched_time_bins == 0:
                return f"Cannot find the time axis ({time_bins} bins) for {channel_name} fluorescence lifetime standard file dimensions: {fluorescence_lifetime_standard_shape}", fov_df
            elif matched_time_bins > 1:
                return f"Ambiguous time axis for {channel_name} fluorescence lifetime standard file dimensions: {fluorescence_lifetime_standard_shape}", fov_df
            else:
                fov_df[f"{channel_name}_fluorescence_lifetime_standard_time_axis"] = fluorescence_lifetime_standard_shape.index(time_bins)
        except Exception as e:  # noqa: BLE001
            return f"Error reading fluorescence lifetime standard file for {channel_name}: {str(e)}", fov_df

    return "", fov_df


def finalize_fov_processing(fov_df, selected_channels, decay_input_type, imaging_modalities, duration, time_bins, folder_path, selected_ch_feature_extractors, fit_free_calibration_method=None, fluorescence_lifetime_standard_lifetime=None):
    """Assign channels, validate standards, then preview + export the metadata.

    Follows the ``(error_msg, result)`` convention: returns ``("", fov_df)`` on
    success, or ``(error_msg, fov_df)`` at the first failing step. Rendering the
    error message is the caller's responsibility.
    """
    # Check and assign channels
    error_msg, fov_df = check_assign_channel_widget(
        fov_df, selected_channels,
        flim_decay_input_type=decay_input_type,
        imaging_modalities=imaging_modalities,
        selected_ch_feature_extractors=selected_ch_feature_extractors,
        duration=duration, time_bins=time_bins
    )
    if error_msg != "":
        return error_msg, fov_df

    # Validate fluorescence lifetime standard file per channel after channel assignment
    if fit_free_calibration_method is not None:
        time_bins = fov_df["time_bins"].iloc[0]
        error_msg, fov_df = validate_fluorescence_lifetime_standard_per_channel(fov_df, selected_channels, fit_free_calibration_method, time_bins, fluorescence_lifetime_standard_lifetime)
        if error_msg != "":
            return error_msg, fov_df

    # Display and export
    preview_metadata_widget(fov_df)
    export_metadata_widget(metadata_df=fov_df, folder_path=folder_path)
    return "", fov_df


def render_fov_metadata_step(col1, col2, ctx):
    """Step 1: select channels + suffixes + folder (col1), then scan/validate/export (col2)."""
    error_msg = ""
    actual_file_suffix = None
    selected_channels = {}
    selected_ch_num_components = {}
    duration = None
    time_bins = None
    laser_rate = None
    folder_path = ""
    fit_free_calibration_method = ctx.fit_free_calibration_method
    fluorescence_lifetime_standard_lifetime = ctx.fluorescence_lifetime_standard_lifetime

    with col1:
        # Tell the user (don't auto-reload) if another tab changed the config since
        # this page loaded, so they can refresh to pick up new channels/extractors.
        notify_on_config_change()
        # show decay input type
        if ctx.has_flim:
            st.write(f"Decay input type: {ctx.decay_input_type}")
        checkbox_cols = st.columns(len(ctx.channel_names))

        for index, (channel_key, channel_name) in enumerate(ctx.channel_names.items()):
            with checkbox_cols[index]:
                has_channel = st.checkbox(f"has {channel_name}", value=True, key=f"has_channel_{channel_key}")
                if has_channel:
                    # have a help text to show the planned features to be extracted
                    with st.expander(f"Feature extractors for {channel_name}", expanded=False):
                        st.write(ctx.selected_ch_feature_extractors[channel_key])
                    selected_channels[channel_key] = channel_name
                    if ctx.ch_num_components[channel_key] != 0 and "prefitted" in ctx.input_types[channel_key]:  # if equals to 0, it means this channel does not have any lifetime fit analysis; only prefitted needs to be specified to get all the files.
                        selected_ch_num_components[channel_name] = st.number_input("No. component", value=ctx.ch_num_components[channel_key], min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis" if index == 0 else None, key=f"num_component_{channel_name}")
                    elif ctx.ch_num_components[channel_key] != 0:  # do not ask now, will ask later when fitting
                        selected_ch_num_components[channel_name] = ctx.ch_num_components[channel_key]
        if len(selected_channels) == 0:
            error_msg = "Please check at least one of the channels"
            st.error(f"{error_msg} {sad_emoji}")
        else:
            if ctx.has_flim:
                duration, time_bins, laser_rate = lifetime_data_config_widget(ctx.selected_ch_feature_extractors, ctx.decay_input_type)
            else:  # for later, we will add other imaging modalities and this will ask for those imaging modality specific config
                duration, time_bins, laser_rate = None, None, None
            if laser_rate is None:
                fit_free_calibration_method = None
            # laser rate is none means there is no fit free analysis
            if fit_free_calibration_method == "Fluorescence Lifetime Standard":
                # Fluorescence lifetime standard file is per-channel and collected via suffixes; only lifetime is shared
                fluorescence_lifetime_standard_lifetime = st.number_input("Fluorescence lifetime standard's lifetime in **ns**", value=fluorescence_lifetime_standard_lifetime, min_value=0.1, max_value=20.0, step=0.1, key="fluorescence_lifetime_standard_lifetime")

            actual_file_suffix, error_msg = load_data_suffix_widget(ctx.input_types, selected_channels, selected_ch_num_components, ctx.selected_ch_feature_extractors)
            if error_msg != "":
                st.error(error_msg)
            else:
                folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the raw data that is needed for the selected data extraction type.", key="fov_metadata_folder_path")
                if folder_path and st.button("Rescan folder", help="Re-read files from disk, ignoring cached results"):
                    clear_folder_scan_caches()
                    st.rerun()

    with col2:
        if error_msg == "":
            # Step 1: Validate folder path
            if not validate_folder_path(folder_path):
                pass  # Error already displayed in function
            else:
                # Step 2: Load and validate FOVs
                fovs = load_and_validate_fovs(folder_path, actual_file_suffix)
                if fovs is None:
                    pass  # Error already displayed in function
                else:
                    # Step 3: Prepare dataframe
                    fov_df = prepare_fov_dataframe(fovs, selected_channels, selected_ch_num_components, ctx)
                    if laser_rate is not None:
                        fov_df["laser_rate"] = laser_rate

                    # Step 4: Finalize processing ( fluorescence lifetime standard validation moved here)
                    error_msg, fov_df = finalize_fov_processing(fov_df, selected_channels, ctx.decay_input_type, ctx.imaging_modalities, duration, time_bins, folder_path, ctx.selected_ch_feature_extractors, fit_free_calibration_method, fluorescence_lifetime_standard_lifetime)
                    if error_msg != "":
                        st.error(f"{error_msg} {sad_emoji}")


# --- Numeric Feature Extraction helpers ------------------------------------
def _load_metadata_df():
    """Load the FOV metadata: prefer the last extracted table in session, else a file upload."""
    metadata_df = None
    if st.session_state["last_extracted_metadata_filepath"] is not None:
        file_path = st.session_state["last_extracted_metadata_filepath"]
        st.info(f"Using the latest extracted metadata file: {file_path}. Refresh the page to use a different file.")
    if st.session_state["last_extracted_metadata"] is not None:
        metadata_df = st.session_state["last_extracted_metadata"]
    else:
        uploaded_file = st.file_uploader("Upload the field of view metadata csv", type=["csv"], help="The metadata file should be from the FOV metadata extraction step.")
        if uploaded_file is not None:
            try:
                metadata_df = pd.read_csv(uploaded_file)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error reading the uploaded CSV file: {e} {sad_emoji}")
                metadata_df = None  # Ensure metadata_df is None if reading fail
    return metadata_df


def _save_or_download_metadata(metadata_df):
    """Save the augmented metadata back to its known path (button-gated), else offer a download."""
    # have a download button to download the metadata file
    if st.session_state["last_extracted_metadata_filepath"] is not None:
        download = st.button("Download updated metadata", width='stretch', help="Download the augmented metadata with the calculated shifts and selected time gates as a CSV file.")
        if download:
            try:
                metadata_df.to_csv(st.session_state["last_extracted_metadata_filepath"], index=False)
                st.success(f"✅ Metadata updated successfully at {st.session_state['last_extracted_metadata_filepath']} {happy_emoji}")
            except PermissionError:
                st.error(f"❌ Cannot save file - it may be open in another program (like Excel). Please close the file and try again. {sad_emoji}")
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ Error saving file: {str(e)} {sad_emoji}")
    else:
        st.download_button(label="Download updated metadata", data=metadata_df.to_csv(index=False), file_name=f"fov_metadata_{time.strftime('%Y%m%d_%H%M%S')}.csv", key=f"download_metadata_{time.time()}", width='stretch', help="Download the augmented metadata with the calculated shifts and selected time gates as a CSV file.")


def _render_shift_controls(metadata_df, metadata_dict):
    """Configure fitting/shift options (col1). Returns the possibly-updated (metadata_df, metadata_dict)."""
    st.success(f"✅ Features to be extracted confirmed. {happy_emoji}")
    # Only relevant when at least one channel is FLIM; not set for intensity-only metadata
    decay_input_type = metadata_dict.get("decay_input_type")
    shift_needed = len(metadata_dict["channels_shift"]) > 0
    shifts_are_present = all(f"{ch}_shift" in metadata_df.columns for ch in metadata_dict["channels_shift"])
    # Defensive reset: if shifts are required but missing, do not allow extraction yet
    if shift_needed and not shifts_are_present and st.session_state.get("shift_ready", False):
        st.session_state["shift_ready"] = False
    if shift_needed and not shifts_are_present:
        # if there are channels to be fitted, show the fitting options: spcimage is already fitted (only for FLIM)
        if decay_input_type is not None and "Lifetime fit" in metadata_dict and len(metadata_dict["Lifetime fit"]) > 0 and "prefitted" not in decay_input_type:
            st.info("Please specify the following fitting options.")
            metadata_dict = fit_options_widget(metadata_dict)
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            metadata_dict["fix_shift"] = st.checkbox(
                "Fix the Shift",
                value=True,
                key="fix_shift_checkbox",
                help="If True, the shift will be fixed for all images. If False, the shift will be estimated for each image."
            )
        with col1_2:
            if st.button("Optimize for Shifts"):
                st.session_state["choosing_shift"] = True
                st.session_state["shift_ready"] = False
                st.rerun()
    else:
        if "fitting_mode" in metadata_df.columns:
            metadata_df["fitting_mode"] = st.selectbox(
                "Fitting Mode",
                ["Hybrid", "Local"],
                index=0,
                key="fitting_mode_update",
                help="Hybrid: global search for initial guess, then local refinement per cell (robust). Local: warm-start on mean decay, then local fit per cell (faster)."
            )
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            if st.button("Confirm and Start", width='stretch'):
                # Update metadata_df in session state
                st.session_state["last_extracted_metadata"] = metadata_df
                st.session_state["choosing_shift"] = False
                st.session_state["shift_ready"] = True
                # Arm the one-shot celebration; _render_run_extraction consumes it
                # once the batch actually produces features.
                st.session_state["celebrate_extraction"] = True
                st.rerun()

        if shift_needed and shifts_are_present:
            with col1_2:
                if st.button("Go back and find shift", width='stretch'):
                    st.session_state["choosing_shift"] = True
                    st.session_state["shift_ready"] = False
                    # remove shift columns from metadata_df in session state
                    for ch in metadata_dict["channels_shift"]:
                        if f"{ch}_shift" in metadata_df.columns:
                            metadata_df = metadata_df.drop(columns=[f"{ch}_shift"])
                    st.session_state["last_extracted_metadata"] = metadata_df
                    st.rerun()
            _save_or_download_metadata(metadata_df)
    return metadata_df, metadata_dict


def _render_choose_shift(metadata_df, metadata_dict, ctx):
    """Per-channel shift optimization (col2); on confirm, persist shifts/time-gates/fit options."""
    channel_shifts = {}
    for channel_name in metadata_dict["channels_shift"]:
        error_msg, shifts = choose_shift_widget(metadata_df, metadata_dict, ctx.fov_name_col, channel_name=channel_name)
        if error_msg != "":
            st.error(f"{error_msg} {sad_emoji}")
        else:
            channel_shifts[channel_name] = shifts
    shift_finished = st.button("Confirm Time Gates (if applicable) and Shift for each channel")
    if shift_finished:
        # write the shift, time gates and fitting options to the metadata file
        for channel_name, shift in channel_shifts.items():
            metadata_df[f"{channel_name}_shift"] = shift
            if "start" in metadata_dict[channel_name]:
                metadata_df[f"{channel_name}_start"] = metadata_dict[channel_name]["start"]
            if "end" in metadata_dict[channel_name]:
                metadata_df[f"{channel_name}_end"] = metadata_dict[channel_name]["end"]
            if "num_components" in metadata_dict[channel_name]:
                metadata_df[f"{channel_name}_num_components"] = metadata_dict[channel_name]["num_components"]
            # Persist any session-level fixed-lifetime overrides back to the metadata CSV
            fixed_lts = metadata_dict[channel_name].get("fixed_lifetimes", {})
            for t_key in ["t1", "t2", "t3"]:
                col = f"{channel_name}_fixed_{t_key}"
                if t_key in fixed_lts:
                    metadata_df[col] = fixed_lts[t_key]  # float or None

        if "fitting_algo" in metadata_dict:
            metadata_df["fitting_algo"] = metadata_dict["fitting_algo"]
        if "fitting_mode" in metadata_dict:
            metadata_df["fitting_mode"] = metadata_dict["fitting_mode"]

        # Store the updated metadata_df in session state so it persists across rerun
        st.session_state["last_extracted_metadata"] = metadata_df
        st.session_state["choosing_shift"] = False
        st.session_state["shift_ready"] = False
        st.rerun()


def _save_or_download_features(single_cell_features, timestamp):
    """Auto-save the single-cell features next to the metadata file, else offer a download."""
    # get the folder path from the file path
    if st.session_state["last_extracted_metadata_filepath"] is not None:
        folder_path = os.path.dirname(st.session_state["last_extracted_metadata_filepath"])
        csv_path = os.path.join(folder_path, f"single_cell_features_{timestamp}.csv")
        # save the features to a csv file automatically
        try:
            single_cell_features.to_csv(csv_path)  # Save the DataFrame
            st.success(f"✅ Single cell features exported successfully to {csv_path} {happy_emoji}")
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Error exporting the single cell features: {str(e)} {sad_emoji}")
    else:
        downloaded = st.download_button(label="Download single cell features as CSV", data=single_cell_features.to_csv(), file_name=f"single_cell_features_{timestamp}.csv")
        if downloaded:
            st.success(f"✅ Single cell features exported successfully to your download folder {happy_emoji}")


def _render_run_extraction(metadata_df, metadata_dict):
    """Run the batch per-FOV extraction (col2) and export the resulting single-cell features."""
    single_cell_features = fov_extraction_widget(metadata_df, metadata_dict)
    if not single_cell_features.empty:
        st.success(f"Fields of view features with ✅ are extracted successfully {happy_emoji}! FOVs with error messages are excluded. The first few rows of the features are shown below.")
        # fov_extraction_widget re-runs the whole batch on every rerun, so this block
        # re-renders whenever the user touches a widget (e.g. the download button
        # below). pop() consumes the flag armed by "Confirm and Start", keeping the
        # animation to once per confirmed run instead of once per interaction.
        if st.session_state.pop("celebrate_extraction", False):
            st.balloons()
        st.write(single_cell_features.head())
        # get the current timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        _save_or_download_features(single_cell_features, timestamp)


def render_numeric_step(col1, col2, ctx):
    """Step 2: load metadata + configure fit/shift (col1); optimize shifts or run extraction (col2)."""
    metadata_df = None
    metadata_dict = None
    with col1:
        metadata_df = _load_metadata_df()
        if metadata_df is not None:
            error_msg, metadata_dict = parse_metadata_file(metadata_df, ctx.fov_name_col)
            if error_msg == "":
                metadata_df, metadata_dict = _render_shift_controls(metadata_df, metadata_dict)
            else:
                st.error(f"Error: {error_msg} {sad_emoji}")

    if metadata_df is None or metadata_dict is None:
        return

    if st.session_state["choosing_shift"]:
        with col2:
            _render_choose_shift(metadata_df, metadata_dict, ctx)
    elif st.session_state["shift_ready"]:
        with col2:
            _render_run_extraction(metadata_df, metadata_dict)


def render_categorical_step(col1, col2, ctx):
    """Step 3: scan a folder of CSVs (col1), then merge + assign categorical labels (col2)."""
    df_folder_path = ""
    delimiter = "_"
    available_dfs = []
    with col1:
        # Categorical features extraction
        df_folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the csv files that you want to assign categories to.")
        delimiter = st.text_input("Field of View Name Delimiter", "_", max_chars=2, help="The delimiter used to split the fov_name column.")
        if df_folder_path != "":
            available_dfs = find_available_dfs_widget(df_folder_path, delimiter)
            if len(available_dfs) > 0:
                st.write(f"Found {len(available_dfs)} available csv files ready to be assigned categories {happy_emoji}:")
                st.write(available_dfs)
            else:
                st.error(f"No available csv files found at {df_folder_path} {sad_emoji}")

    with col2:
        if df_folder_path != "" and len(available_dfs) > 0:
            combined_df, available_categories = check_and_merge_df_widget(available_dfs)
            map_categories_to_labels_widget(available_categories, combined_df, delimiter, df_folder_path)


# --- Page controller -------------------------------------------------------
STEP_RENDERERS = {
    STEP_FOV: render_fov_metadata_step,
    STEP_NUMERIC: render_numeric_step,
    STEP_CATEGORICAL: render_categorical_step,
}

st.set_page_config(layout="wide", page_icon="🔬")
# Render the top menu
render_top_menu()
init_session_state()
st.title("Data Extraction")

ctx = build_context()
# A blank/unconfigured active profile (e.g. one just created in the Configuration
# page's sidebar but never saved with "Update Configuration") has no channels.
# Stop with guidance here instead of crashing downstream on st.columns(0).
if not ctx.channel_names:
    st.warning(
        f"The active configuration profile **'{get_current_profile_name()}'** has not "
        "been configured yet. Please go to **Home / Configuration** page, "
        "configure this profile, then click "
        "**Update Configuration** — and come back here."
    )
    st.stop()

col1, col2 = st.columns([0.4, 1])
with col1:
    # first select the step to perform
    selected_step = st.radio(
        "Select a step to perform",
        STEPS,
        index=0,
        help="FOV Metadata Extraction: Extracts metadata from the field of views. Numeric Feature Extraction: Extracts single cell numeric features from the FOVs. Categorical Feature Extraction: Extracts categorical features from the FOVs. \n ",
    )

STEP_RENDERERS[selected_step](col1, col2, ctx)
